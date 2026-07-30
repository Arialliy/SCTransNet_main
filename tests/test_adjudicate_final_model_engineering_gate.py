from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import adjudicate_final_model_engineering_gate as subject
from experiments import (
    evaluate_final_model_engineering_replication_pd_fa as evaluation_core,
)
from experiments import final_model_replication_exact_core as core
from experiments import final_model_replication_seed_contract as seeds
from experiments import summarize_final_model_engineering_replication as summary_core
from experiments import watch_final_model_engineering_replication as watcher


SHA = "a" * 64
VALID_PIXELS = 133 * 256 * 256
VALIDATION_IDS = tuple(
    f"validation_{index:03d}" for index in range(133)
)


def metric_point(
    *,
    threshold: float = 0.5,
    pd: float = 188 / 189,
    fa: float = 2e-6,
    miou: float = 0.93,
    tiny_pd: float = 1.0,
    false_objects: int = 2,
) -> dict[str, float | int]:
    return {
        "threshold": threshold,
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
        "valid_pixel_count": VALID_PIXELS,
    }


def checkpoint_metrics(point: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: value
        for key, value in point.items()
        if key != "threshold"
    }


def result_payload(
    *,
    point: dict[str, float | int],
) -> dict[str, object]:
    return {
        "schema": evaluation_core.RESULT_SCHEMA,
        "fixed_threshold_0_5": copy.deepcopy(point),
        "best_points_under_fa_budget": {
            key: {
                **copy.deepcopy(point),
                "threshold": min(1.0, 0.5 + index * 0.01),
                "fa": min(float(point["fa"]), budget),
            }
            for index, (key, budget) in enumerate(
                zip(
                    evaluation_core.BUDGET_KEYS,
                    evaluation_core.FA_BUDGETS,
                )
            )
        },
    }


def launcher_log_text(run: dict[str, object]) -> str:
    checkpoints = {
        str(record["filename"]): record
        for record in run["checkpoints"]
    }
    lines = [
        (
            f"START variant={run['variant']} mode=parent_warm_start "
            "completed=0 next=1 device=cuda:0"
        ),
        *[
            (
                f"EPOCH {epoch:03d}/800 "
                "total=0.003000 seg=0.002000 surv=0.200000 "
                "mIoU=0.930000 Pd=0.994709 Fa=0.00000200"
            )
            for epoch in range(1, 801)
        ],
        (
            f"COMPLETE variant={run['variant']} "
            f"bestPdEpoch={checkpoints['best.pth.tar']['epoch']} "
            f"bestMiouEpoch={checkpoints['best_miou.pth.tar']['epoch']}"
        ),
        f"OUTPUT {run['run_directory']}",
    ]
    return "\n".join(lines) + "\n"


class EngineeringGateTests(unittest.TestCase):
    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "output_root": root / "results",
            "source_lock_path": root / "source-lock.json",
            "seed_contract_path": root / "seed-contract.json",
            "manifest_directory": root / "manifests",
            "summary_path": root / "summary.json",
        }

    def _request(
        self,
        output_root: Path,
        *,
        trajectory_seed: int,
        arm: str,
        checkpoint_filename: str,
        point: dict[str, float | int],
    ) -> evaluation_core.CheckpointEvaluationRequest:
        definition = core.arm_definition(arm)
        selection_role, checkpoint_role = {
            filename: (selection, role)
            for filename, selection, role in evaluation_core.CHECKPOINT_SPECS
        }[checkpoint_filename]
        run_directory = watcher.run_directory(
            output_root,
            trajectory_seed,
            arm,
        )
        return evaluation_core.CheckpointEvaluationRequest(
            trajectory_seed=trajectory_seed,
            arm=arm,
            variant=definition.variant,
            run_directory=run_directory,
            run_identity={
                "run_id": (
                    f"seed-{trajectory_seed}-{arm}-{checkpoint_filename}"
                )
            },
            seed_contract_path=output_root / "seed-contract.json",
            seed_contract_sha256=SHA,
            child_manifest_path=output_root / "child-manifest.json",
            child_manifest_sha256=SHA,
            source_lock_path=output_root / "source-lock.json",
            source_lock_sha256=SHA,
            protocol_sha256=SHA,
            split_sha256=SHA,
            summary_sha256=SHA,
            selection_role=selection_role,
            checkpoint_filename=checkpoint_filename,
            checkpoint_path=run_directory / checkpoint_filename,
            checkpoint_role=checkpoint_role,
            checkpoint_epoch=17,
            checkpoint_sha256=SHA,
            checkpoint_validation_metrics=checkpoint_metrics(point),
            metrics_sha256=SHA,
            training_data_sha256=SHA,
            normalization_sha256=SHA,
            validation_split_sha256=(
                evaluation_core.statistics_cache
                .validation_identifier_sha256(VALIDATION_IDS)
            ),
            validation_ids=VALIDATION_IDS,
        )

    def _complete_fixture(
        self,
        root: Path,
    ) -> tuple[
        dict[str, Path],
        dict[str, object],
        list[evaluation_core.CheckpointEvaluationRequest],
    ]:
        paths = self._paths(root)
        requests: list[
            evaluation_core.CheckpointEvaluationRequest
        ] = []
        summary_runs: list[dict[str, object]] = []
        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
            for arm in core.SUPPORTED_ARMS:
                is_d = arm == core.ARM_D
                fixed = metric_point(
                    pd=188 / 189,
                    fa=2e-6 if is_d else 3e-6,
                    miou=0.93 if is_d else 0.92,
                    tiny_pd=1.0,
                    false_objects=2 if is_d else 3,
                )
                checkpoints: list[dict[str, object]] = []
                for checkpoint_filename, selection_role, checkpoint_role in (
                    evaluation_core.CHECKPOINT_SPECS
                ):
                    request = self._request(
                        paths["output_root"],
                        trajectory_seed=trajectory_seed,
                        arm=arm,
                        checkpoint_filename=checkpoint_filename,
                        point=fixed,
                    )
                    requests.append(request)
                    checkpoints.append(
                        {
                            "selection_role": selection_role,
                            "filename": checkpoint_filename,
                            "path": str(
                                (
                                    watcher.run_directory(
                                        paths["output_root"],
                                        trajectory_seed,
                                        arm,
                                    )
                                    / checkpoint_filename
                                ).resolve()
                            ),
                            "sha256": request.checkpoint_sha256,
                            "epoch": request.checkpoint_epoch,
                            "checkpoint_role": checkpoint_role,
                            "metrics": copy.deepcopy(
                                request.checkpoint_validation_metrics
                            ),
                        }
                    )
                    request.planned_output_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    request.planned_output_path.write_text(
                        json.dumps(
                            result_payload(point=fixed),
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                summary_runs.append(
                    {
                        "arm": arm,
                        "variant": core.arm_definition(arm).variant,
                        "trajectory_seed": trajectory_seed,
                        "run_directory": str(
                            watcher.run_directory(
                                paths["output_root"],
                                trajectory_seed,
                                arm,
                            ).resolve()
                        ),
                        "summary_sha256": SHA,
                        "seed_contract_sha256": SHA,
                        "source_lock_sha256": SHA,
                        "child_initialization_manifest_sha256": SHA,
                        "checkpoints": checkpoints,
                    }
                )
        summary = {
            "schema": summary_core.SCHEMA,
            "status": "complete",
            "scope": "fixed_parent_engineering_b_d_only",
            "run_count": 4,
            "checkpoint_count": 8,
            "checkpoint_selection": {
                "primary": "each_arm_own_best_miou",
                "secondary": "each_arm_own_best_pd",
                "cross_arm_shared_checkpoint_epoch_required": False,
            },
            "fixed_threshold": 0.5,
            "pd_fa_sweep_complete": False,
            "official_test_accessed": False,
            "runs": summary_runs,
            "paired_best_miou_comparisons": [],
            "claim_boundary": {
                "engineering_replication_complete": True,
                "paper_stability_supported": False,
                "full_pipeline_stability_supported": False,
            },
        }
        expected = subject._expected_artifacts(
            output_root=paths["output_root"],
            source_lock_path=paths["source_lock_path"],
            seed_contract_path=paths["seed_contract_path"],
            manifest_directory=paths["manifest_directory"],
            summary_path=paths["summary_path"],
        )
        run_index = {
            (int(run["trajectory_seed"]), str(run["arm"])): run
            for run in summary_runs
        }
        for record in expected:
            path = Path(record["path"])
            if path == paths["summary_path"] or path.name.startswith(
                "pd_fa_sweep_"
            ):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if str(record["role"]).endswith("_launcher_log"):
                parts = path.stem.split("_")
                run = run_index[(int(parts[1]), parts[2])]
                path.write_text(
                    launcher_log_text(run),
                    encoding="utf-8",
                )
            else:
                path.write_bytes(b"fixture")
        paths["summary_path"].write_bytes(
            summary_core.canonical_json_bytes(summary)
        )
        return paths, summary, requests

    def _run_complete(
        self,
        paths: dict[str, Path],
        summary: dict[str, object],
        requests: list[
            evaluation_core.CheckpointEvaluationRequest
        ],
        *,
        validator: object | None = None,
    ) -> dict[str, object]:
        def preflight(**kwargs):
            return tuple(
                request
                for request in requests
                if request.trajectory_seed == kwargs["trajectory_seed"]
                and request.arm == kwargs["arm"]
            )

        result_validator = (
            validator
            if validator is not None
            else lambda payload, request: copy.deepcopy(dict(payload))
        )
        with (
            mock.patch.object(
                subject.summary_core,
                "build_summary",
                return_value=copy.deepcopy(summary),
            ),
            mock.patch.object(
                subject.evaluation_core,
                "preflight_completed_run",
                side_effect=preflight,
            ),
            mock.patch.object(
                subject.evaluation_core,
                "assemble_evaluation_plan",
                return_value={"request_count": 8},
            ),
            mock.patch.object(
                subject.evaluation_core,
                "validate_checkpoint_local_result",
                side_effect=result_validator,
            ),
        ):
            return subject.adjudicate(**paths)

    def test_inventory_sweep_paths_match_evaluator_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            paths, _, requests = self._complete_fixture(
                Path(directory_text)
            )
            expected = subject._expected_artifacts(
                output_root=paths["output_root"],
                source_lock_path=paths["source_lock_path"],
                seed_contract_path=paths["seed_contract_path"],
                manifest_directory=paths["manifest_directory"],
                summary_path=paths["summary_path"],
            )
            inventory_sweeps = {
                Path(record["path"])
                for record in expected
                if str(record["role"]).endswith("_sweep")
            }
            request_sweeps = {
                request.planned_output_path for request in requests
            }
        self.assertEqual(inventory_sweeps, request_sweeps)
        self.assertFalse(
            any(
                "pd_fa_sweep_pd_fa_sweep_" in path.name
                for path in inventory_sweeps
            )
        )

    def test_missing_evidence_returns_pending_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            payload = subject.adjudicate(
                **self._paths(Path(directory_text))
            )
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["decision"], "ENGINEERING_GATE_PENDING")
        self.assertTrue(payload["missing_artifacts"])
        self.assertNotIn(
            "fixed_threshold_and_budget_comparisons",
            payload,
        )
        self.assertFalse(
            payload["claim_boundary"]["stability_claim_supported"]
        )

    def test_complete_evidence_passes_s_e_but_not_m_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            paths, summary, requests = self._complete_fixture(
                Path(directory_text)
            )
            payload = self._run_complete(paths, summary, requests)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            payload["decision"],
            "ENGINEERING_GATE_S_E_PASS",
        )
        self.assertTrue(payload["gates"]["S-E"]["passed"])
        self.assertEqual(
            payload["gates"]["M-train"]["status"],
            "insufficient_evidence",
        )
        self.assertIsNone(payload["gates"]["M-train"]["passed"])
        self.assertFalse(
            payload["gates"]["M-train"][
                "aggregate_point_estimates_used_as_ci"
            ]
        )
        self.assertEqual(
            len(payload["fixed_threshold_and_budget_comparisons"]),
            4,
        )
        self.assertEqual(
            payload["evidence"]["data_contract"]["validation_count"],
            133,
        )
        self.assertEqual(payload["evidence"]["launcher_log_count"], 4)
        self.assertEqual(
            len(payload["evidence"]["launcher_logs"]),
            4,
        )
        for log in payload["evidence"]["launcher_logs"]:
            self.assertEqual(log["status"], "verified_complete")
            self.assertEqual(log["epoch_line_count"], 800)
            self.assertEqual(
                log["epoch_sequence"],
                "exactly_1_through_800_once",
            )
            self.assertGreater(log["byte_count"], 0)
            self.assertEqual(len(log["sha256"]), 64)
        for comparison in payload[
            "fixed_threshold_and_budget_comparisons"
        ]:
            self.assertEqual(
                set(comparison["fa_budget_envelopes"]),
                set(evaluation_core.BUDGET_KEYS),
            )
            self.assertIn(
                "unmatched_predicted_object_count",
                comparison["d_minus_b"],
            )
        screen = payload["engineering_performance_screen"]
        self.assertEqual(screen["seed_direction_met_count"], 2)
        self.assertFalse(screen["uses_paired_image_level_ci"])
        self.assertFalse(screen["establishes_gate_m_train"])
        self.assertFalse(
            payload["claim_boundary"]["paper_core_established"]
        )
        self.assertFalse(
            payload["claim_boundary"]["stability_claim_supported"]
        )

    def test_present_but_unvalidated_sweep_returns_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            paths, summary, requests = self._complete_fixture(
                Path(directory_text)
            )

            def reject(payload, request):
                raise evaluation_core.EngineeringEvaluationError(
                    "foreign checkpoint-local point"
                )

            payload = self._run_complete(
                paths,
                summary,
                requests,
                validator=reject,
            )
        self.assertEqual(payload["status"], "invalid")
        self.assertEqual(payload["decision"], "ENGINEERING_GATE_INVALID")
        self.assertIn("foreign checkpoint-local point", payload["errors"][0])
        self.assertFalse(payload["gates"]["S-E"]["engineering_replication_complete"])

    def test_symlink_input_is_invalid_not_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            root = Path(directory_text)
            paths = self._paths(root)
            target = root / "actual-source-lock.json"
            target.write_text("{}\n", encoding="utf-8")
            paths["source_lock_path"].symlink_to(target)
            payload = subject.adjudicate(**paths)
        self.assertEqual(payload["status"], "invalid")
        self.assertTrue(
            any("symlink" in error for error in payload["errors"])
        )

    def test_launcher_log_must_be_nonempty_and_complete(self) -> None:
        mutations = {
            "empty": lambda _text: "",
            "missing_epoch": lambda text: text.replace(
                (
                    "EPOCH 400/800 total=0.003000 seg=0.002000 "
                    "surv=0.200000 mIoU=0.930000 Pd=0.994709 "
                    "Fa=0.00000200\n"
                ),
                "",
                1,
            ),
            "duplicate_epoch": lambda text: text.replace(
                "EPOCH 400/800 ",
                "EPOCH 399/800 ",
                1,
            ),
            "wrong_complete": lambda text: text.replace(
                "bestPdEpoch=17",
                "bestPdEpoch=18",
                1,
            ),
            "wrong_output": lambda text: text.replace(
                "\nOUTPUT ",
                "\nOUTPUT /foreign/",
                1,
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory_text:
                paths, summary, requests = self._complete_fixture(
                    Path(directory_text)
                )
                log = (
                    paths["output_root"]
                    / "logs"
                    / f"seed_{seeds.ENGINEERING_TRAJECTORY_SEEDS[0]}_b.log"
                )
                log.write_text(
                    mutate(log.read_text(encoding="utf-8")),
                    encoding="utf-8",
                )
                payload = self._run_complete(paths, summary, requests)
                self.assertEqual(payload["status"], "invalid")
                self.assertEqual(
                    payload["decision"],
                    "ENGINEERING_GATE_INVALID",
                )
                self.assertTrue(payload["errors"])

    def test_write_once_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            output = Path(directory_text) / "gate.json"
            payload = {"schema": subject.SCHEMA, "status": "complete"}
            subject.write_once(output, payload)
            with self.assertRaises(FileExistsError):
                subject.write_once(output, payload)


if __name__ == "__main__":
    unittest.main()
