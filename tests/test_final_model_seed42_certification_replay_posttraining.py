from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments import (
    final_model_seed42_certification_replay_posttraining as post,
)


def metric_point(
    *,
    pd: float = 1.0,
    fa: float = 0.0,
    miou: float = 0.9,
    tiny_pd: float = 1.0,
    false_objects: int = 0,
) -> dict[str, object]:
    return {
        "threshold": 0.5,
        "pd": pd,
        "matched_target_count": round(pd * 189),
        "target_count": 189,
        "fa": fa,
        "miou": miou,
        "tiny_pd": tiny_pd,
        "matched_tiny_target_count": round(tiny_pd * 39),
        "tiny_target_count": 39,
        "false_objects_per_image": false_objects / 133,
        "unmatched_predicted_object_count": false_objects,
        "valid_pixel_count": 8716288,
    }


def result_payload(point: dict[str, object]) -> dict[str, object]:
    return {
        "fixed_threshold_0_5": point,
        "best_points_under_fa_budget": {
            key: dict(point, threshold=0.75)
            for key in post.evaluator.BUDGET_KEYS
        },
    }


class Seed42ReplayPosttrainingContractTests(unittest.TestCase):
    def test_contract_is_exactly_one_seed_two_arms_two_checkpoints(self) -> None:
        with mock.patch.object(
            post,
            "_evaluation_source_binding",
            return_value={"adapter": {"path": "/x", "sha256": "a" * 64}},
        ):
            contract = post._evaluator_contract()
        self.assertEqual(contract["trajectory_seeds"], [42])
        self.assertEqual(contract["excluded_seeds"], [3407, 426780603])
        self.assertEqual(contract["arms"], ["b", "d"])
        self.assertEqual(contract["expected_run_count"], 2)
        self.assertEqual(contract["expected_sweep_count"], 4)
        self.assertEqual(
            [item["filename"] for item in contract["checkpoint_policy"]],
            ["best_miou.pth.tar", "best.pth.tar"],
        )
        self.assertEqual(contract["fixed_threshold"], 0.5)
        self.assertFalse(contract["cross_checkpoint_point_pooling"])
        self.assertFalse(contract["old_seed42_results_accepted"])
        self.assertFalse(contract["paper_core_established"])
        self.assertFalse(contract["stability_claim_supported"])

    def test_canonical_matrix_order_is_seed42_b_then_d(self) -> None:
        self.assertEqual(
            post._canonical_request_keys(),
            (
                (42, "b", "best_miou.pth.tar"),
                (42, "b", "best.pth.tar"),
                (42, "d", "best_miou.pth.tar"),
                (42, "d", "best.pth.tar"),
            ),
        )

    def test_old_and_non_replay_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            post.Seed42ReplayPosttrainingError,
            "outside the new replay",
        ):
            post._assert_new_replay_path(
                Path(
                    "/home/ly/SCTransNet_main/experiments/results/"
                    "tpd_ner_v4_survival_exact_v1/NUDT-SIRST/tss_on/"
                    "seed_42_formal800_tss"
                ),
                "legacy",
            )
        with self.assertRaisesRegex(
            post.Seed42ReplayPosttrainingError,
            "forbidden seed 3407",
        ):
            post._assert_new_replay_path(
                post.DEFAULT_OUTPUT_ROOT / "seed_3407_b",
                "supplement",
            )
        with self.assertRaisesRegex(
            post.Seed42ReplayPosttrainingError,
            "forbidden seed 426780603",
        ):
            post._assert_new_replay_path(
                post.DEFAULT_OUTPUT_ROOT / "seed_426780603_d",
                "cancelled",
            )

    def test_dry_run_reports_progress_without_gpu_or_writes(self) -> None:
        fake_inputs = {
            arm: SimpleNamespace(
                definition=SimpleNamespace(
                    arm=arm,
                    variant="tss_on" if arm == "b" else "tss_qfg",
                ),
                trajectory_seed=42,
            )
            for arm in ("b", "d")
        }
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            run_dirs = {
                arm: directory / f"seed_42_{arm}"
                for arm in ("b", "d")
            }
            for path in run_dirs.values():
                path.mkdir()
                (path / "metrics.jsonl").write_text(
                    json.dumps(
                        {
                            "epoch": 1,
                            "pd": 1.0,
                            "fa": 0.0,
                            "miou": 0.9,
                            "tiny_pd": 1.0,
                            "false_objects_per_image": 0.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            with mock.patch.object(
                post,
                "_inputs",
                side_effect=lambda arm, **_: fake_inputs[arm],
            ), mock.patch.object(
                post.replay_core,
                "run_directory",
                side_effect=lambda inputs: run_dirs[inputs.definition.arm],
            ), mock.patch.object(
                post.replay_core,
                "resolve_initialization_mode",
                return_value="--exact-resume",
            ):
                payload = post.dry_run_payload()
        self.assertEqual(
            payload["status"],
            "waiting_for_seed42_replay_training",
        )
        self.assertFalse(payload["gpu_used"])
        self.assertFalse(payload["gpu_command_launched"])
        self.assertFalse(payload["persistent_artifact_written"])
        self.assertEqual(payload["planned_sweep_count"], 4)
        self.assertEqual([run["completed_epochs"] for run in payload["runs"]], [1, 1])


class Seed42ReplayPosttrainingEvidenceTests(unittest.TestCase):
    def test_build_manifest_preserves_four_real_result_paths(self) -> None:
        paths = [
            post.DEFAULT_OUTPUT_ROOT / f"run_{index}" / "result.json"
            for index in range(4)
        ]
        reused = {
            "schema": post.evaluator.MANIFEST_SCHEMA,
            "scope": "fixed_parent_engineering_b_d_only",
            "result_count": 4,
            "expected_result_count": 4,
            "paired_checkpoint_group_count": 2,
            "gate_m_train_image_level_inputs_ready": True,
            "results": [
                {
                    "trajectory_seed": 42,
                    "result_path": str(path),
                }
                for path in paths
            ],
        }
        with mock.patch.object(
            post.evaluator,
            "build_results_manifest",
            return_value=reused,
        ), mock.patch.object(
            post.evaluator,
            "assemble_evaluation_plan",
            return_value={},
        ):
            manifest = post.build_manifest(())
        self.assertEqual(manifest["schema"], post.MANIFEST_SCHEMA)
        self.assertEqual(manifest["scope"], post.SCOPE)
        self.assertEqual(manifest["result_count"], 4)
        self.assertEqual(manifest["paired_checkpoint_group_count"], 2)
        self.assertEqual(
            [item["trajectory_seed"] for item in manifest["results"]],
            [42, 42, 42, 42],
        )

    def test_paired_normalization_cannot_raise_claims(self) -> None:
        reused = {
            "schema": "old",
            "scope": "old",
            "status": "complete",
            "decision": "old",
            "engineering_paired_route_met": True,
            "per_seed_checkpoint_policy_results": [
                {
                    "trajectory_seed": 42,
                    "seed_role": "wrong",
                    "selection_role": post.PRIMARY_SELECTION_ROLE,
                }
            ],
            "cache_compatibility": {
                "all_eight_cache_targets_identical": True,
                "all_eight_cache_image_ids_identical": True,
                "all_eight_cache_shapes_identical": True,
            },
        }
        with mock.patch.object(
            post.paired_core,
            "analyze",
            return_value=reused,
        ):
            paired = post.analyze_paired(post.DEFAULT_MANIFEST)
        self.assertEqual(paired["schema"], post.PAIRED_SCHEMA)
        self.assertEqual(
            paired["decision"],
            "SEED42_REPLAY_PAIRED_MIOU_ROUTE_MET",
        )
        self.assertTrue(paired["seed42_replay_paired_route_met"])
        self.assertFalse(paired["establishes_gate_m_train"])
        self.assertFalse(
            paired["claim_boundary"]["paper_core_established"]
        )
        self.assertFalse(
            paired["claim_boundary"]["stability_claim_supported"]
        )
        self.assertIn(
            "all_four_cache_targets_identical",
            paired["cache_compatibility"],
        )
        self.assertNotIn(
            "all_eight_cache_targets_identical",
            paired["cache_compatibility"],
        )

    def test_gate_reports_both_checkpoint_roles_and_all_metrics(self) -> None:
        requests = [
            SimpleNamespace(arm=arm, checkpoint_filename=filename)
            for arm in ("b", "d")
            for filename in ("best_miou.pth.tar", "best.pth.tar")
        ]
        b = result_payload(metric_point(miou=0.90, false_objects=2))
        d = result_payload(metric_point(miou=0.91, false_objects=1))
        payloads = {
            ("b", "best_miou.pth.tar"): b,
            ("b", "best.pth.tar"): b,
            ("d", "best_miou.pth.tar"): d,
            ("d", "best.pth.tar"): d,
        }
        paired = {
            "status": "complete",
            "per_seed_checkpoint_policy_results": [
                {
                    "selection_role": post.PRIMARY_SELECTION_ROLE,
                    "paired_image_bootstrap": {
                        "miou_route_delta_0": {
                            "name": "MIOU_ROUTE",
                            "criteria": {
                                "pd_noninferior_delta_0": True,
                                "tiny_pd_noninferior_delta_0": True,
                                "false_objects_noninferior_delta_0": True,
                                "miou_superior_delta_0": True,
                                "fa_noninferior_delta_0": True,
                            },
                            "met": True,
                        }
                    },
                }
            ],
        }

        def load_result(request):
            return payloads[(request.arm, request.checkpoint_filename)], "a" * 64

        with mock.patch.object(
            post.evaluator,
            "assemble_evaluation_plan",
            return_value={},
        ), mock.patch.object(
            post.evaluator,
            "load_completed_result",
            side_effect=load_result,
        ), mock.patch.object(
            post,
            "_validate_replay_manifest_for_paired",
            return_value=({}, {}),
        ), mock.patch.object(
            post,
            "_load_canonical_object",
            return_value={},
        ), mock.patch.object(
            post,
            "_load_and_validate_paired",
            return_value=paired,
        ), mock.patch.object(
            post,
            "_sha256_file",
            return_value="a" * 64,
        ):
            gate = post.adjudicate_gate(
                requests,
                manifest_path=Path("/new/manifest.json"),
                paired_path=Path("/new/paired.json"),
            )
        self.assertEqual(
            gate["decision"],
            "SEED42_REPLAY_ENGINEERING_COMPLETE_MIOU_ROUTE_MET",
        )
        self.assertEqual(
            [item["selection_role"] for item in gate["fixed_threshold_and_budget_comparisons"]],
            [post.PRIMARY_SELECTION_ROLE, post.SECONDARY_SELECTION_ROLE],
        )
        primary = gate["fixed_threshold_and_budget_comparisons"][0]
        for name in (
            "pd",
            "fa",
            "miou",
            "tiny_pd",
            "false_objects_per_image",
        ):
            self.assertIn(name, primary["b"])
            self.assertIn(name, primary["d"])
            self.assertIn(name, primary["d_minus_b"])
        self.assertIsNone(gate["gates"]["M-train"]["passed"])
        self.assertFalse(
            gate["claim_boundary"]["paper_core_established"]
        )
        self.assertFalse(
            gate["claim_boundary"]["stability_claim_supported"]
        )

    def test_closure_is_write_once_and_claims_remain_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            paths = {
                name: directory / f"{name}.json"
                for name in ("summary", "manifest", "paired", "gate")
            }
            for name, path in paths.items():
                value = {
                    "schema": post.GATE_SCHEMA if name == "gate" else name,
                    "status": "complete",
                }
                if name == "gate":
                    value["decision"] = "DONE"
                path.write_bytes(post._canonical_json_bytes(value))
            closure = post.build_closure(
                summary_path=paths["summary"],
                manifest_path=paths["manifest"],
                paired_path=paths["paired"],
                gate_path=paths["gate"],
            )
            output = directory / "closure.json"
            first = post._write_once(output, closure, "closure")
            second = post._write_once(output, closure, "closure")
        self.assertEqual(first, second)
        self.assertEqual(closure["trajectory_seeds"], [42])
        self.assertEqual(closure["sweep_count"], 4)
        self.assertFalse(closure["old_seed42_results_used"])
        self.assertFalse(closure["paper_core_established"])
        self.assertFalse(closure["stability_claim_supported"])

    def test_strict_verifier_rebuilds_every_evidence_layer(self) -> None:
        summary = {"schema": post.SUMMARY_SCHEMA}
        manifest = {"schema": post.MANIFEST_SCHEMA}
        paired = {
            "schema": post.PAIRED_SCHEMA,
            "claim_boundary": {
                "paper_core_established": False,
                "stability_claim_supported": False,
            },
        }
        gate = {
            "schema": post.GATE_SCHEMA,
            "claim_boundary": {
                "paper_core_established": False,
                "stability_claim_supported": False,
            },
        }
        closure = {
            "schema": post.CLOSURE_SCHEMA,
            "paper_core_established": False,
            "stability_claim_supported": False,
        }
        stored = {
            "summary": summary,
            "manifest": manifest,
            "paired": paired,
            "gate": gate,
            "closure": closure,
        }

        def load(path, _label):
            return stored[Path(path).stem]

        with mock.patch.object(
            post,
            "collect_requests",
            return_value=(),
        ), mock.patch.object(
            post,
            "_load_canonical_object",
            side_effect=load,
        ), mock.patch.object(
            post,
            "build_summary",
            return_value=summary,
        ) as build_summary, mock.patch.object(
            post,
            "build_manifest",
            return_value=manifest,
        ) as build_manifest, mock.patch.object(
            post,
            "analyze_paired",
            return_value=paired,
        ) as analyze, mock.patch.object(
            post,
            "adjudicate_gate",
            return_value=gate,
        ) as adjudicate, mock.patch.object(
            post,
            "build_closure",
            return_value=closure,
        ) as build_closure, mock.patch.object(
            post,
            "_sha256_file",
            return_value="a" * 64,
        ):
            verified = post.verify_complete_closure(
                summary_path=Path("summary"),
                manifest_path=Path("manifest"),
                paired_path=Path("paired"),
                gate_path=Path("gate"),
                closure_path=Path("closure"),
            )
        self.assertEqual(verified["status"], "verified_complete")
        self.assertFalse(verified["paper_core_established"])
        self.assertFalse(verified["stability_claim_supported"])
        build_summary.assert_called_once()
        build_manifest.assert_called_once()
        analyze.assert_called_once()
        adjudicate.assert_called_once()
        build_closure.assert_called_once()

    def test_launcher_dry_run_precedes_all_gpu_exports(self) -> None:
        launcher = (
            post.REPO_ROOT
            / "experiments/"
            "run_final_model_seed42_certification_replay_posttraining_2x5090.sh"
        ).read_text(encoding="utf-8")
        dry_run_block = launcher.split('if [[ "$mode" == "--dry-run" ]]', 1)[1]
        dry_run_body = dry_run_block.split("fi", 1)[0]
        self.assertIn("CUDA_VISIBLE_DEVICES=", dry_run_body)
        self.assertIn("--dry-run", dry_run_body)
        self.assertNotIn("gpu2_uuid", dry_run_body)
        self.assertNotIn("gpu3_uuid", dry_run_body)
        self.assertNotIn("3407", launcher)
        self.assertNotIn("426780603", launcher)
        self.assertIn("launch_arm b 2", launcher)
        self.assertIn("launch_arm d 3", launcher)
        self.assertLess(
            launcher.index('flock -n 9'),
            launcher.index('--write-summary'),
        )


if __name__ == "__main__":
    unittest.main()
