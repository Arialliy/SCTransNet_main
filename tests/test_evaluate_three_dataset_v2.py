from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments import evaluate_three_dataset_v2 as subject


def point(
    *,
    threshold: float,
    pd: float,
    fa: float,
    predicted_objects: int,
    miou: float = 0.8,
    niou: float = 0.8,
    tiny_pd: float | None = 0.5,
) -> dict[str, object]:
    return {
        "threshold": threshold,
        "pd": pd,
        "fa": fa,
        "predicted_object_count": predicted_objects,
        "miou": miou,
        "niou": niou,
        "tiny_pd": tiny_pd,
    }


class PdAtFaBudgetTest(unittest.TestCase):
    def test_empty_endpoint_is_selected_only_without_nonempty_feasible(self) -> None:
        points = [
            point(
                threshold=0.5,
                pd=0.9,
                fa=2e-6,
                predicted_objects=10,
            ),
            point(
                threshold=1.0,
                pd=0.0,
                fa=0.0,
                predicted_objects=0,
                miou=0.0,
                niou=0.0,
                tiny_pd=0.0,
            ),
        ]
        selected = subject.pd_at_fa_budget(points, 1e-6)
        self.assertEqual(selected["budget"], 1e-6)
        self.assertEqual(selected["pd_at_fa_budget"], 0.0)
        self.assertEqual(selected["fa_at_selected_point"], 0.0)
        self.assertEqual(selected["selected_threshold"], 1.0)
        self.assertIs(selected["selected_point_is_empty"], True)
        self.assertIs(selected["registered_grid_nonempty_feasible"], False)
        self.assertIsNone(selected["best_nonempty_point"])

    def test_nonempty_feasible_point_takes_precedence_over_empty(self) -> None:
        nonempty = point(
            threshold=0.8,
            pd=0.75,
            fa=0.5e-6,
            predicted_objects=7,
        )
        points = [
            nonempty,
            point(
                threshold=1.0,
                pd=0.0,
                fa=0.0,
                predicted_objects=0,
                miou=0.0,
                niou=0.0,
                tiny_pd=0.0,
            ),
        ]
        selected = subject.pd_at_fa_budget(points, 1e-6)
        self.assertEqual(selected["pd_at_fa_budget"], 0.75)
        self.assertEqual(selected["fa_at_selected_point"], 0.5e-6)
        self.assertEqual(selected["selected_threshold"], 0.8)
        self.assertIs(selected["selected_point_is_empty"], False)
        self.assertIs(selected["registered_grid_nonempty_feasible"], True)
        self.assertEqual(selected["best_nonempty_point"], nonempty)

    def test_nonempty_pd_zero_still_counts_as_nonempty_feasible(self) -> None:
        points = [
            point(
                threshold=0.99,
                pd=0.0,
                fa=0.0,
                predicted_objects=1,
                miou=0.0,
                niou=0.0,
                tiny_pd=0.0,
            ),
            point(
                threshold=1.0,
                pd=0.0,
                fa=0.0,
                predicted_objects=0,
                miou=0.0,
                niou=0.0,
                tiny_pd=0.0,
            ),
        ]
        selected = subject.pd_at_fa_budget(points, 0.0)
        self.assertEqual(selected["selected_threshold"], 0.99)
        self.assertIs(selected["selected_point_is_empty"], False)
        self.assertIs(selected["registered_grid_nonempty_feasible"], True)

    def test_malformed_or_missing_threshold_one_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one threshold=1.0"):
            subject.pd_at_fa_budget(
                [
                    point(
                        threshold=0.5,
                        pd=1.0,
                        fa=0.0,
                        predicted_objects=1,
                    )
                ],
                1e-6,
            )
        with self.assertRaisesRegex(ValueError, "must be an empty-prediction"):
            subject.pd_at_fa_budget(
                [
                    point(
                        threshold=1.0,
                        pd=0.1,
                        fa=0.0,
                        predicted_objects=1,
                    )
                ],
                1e-6,
            )


class ThresholdSeparationTest(unittest.TestCase):
    def test_contract_freezes_half_for_checkpoint_and_lambda(self) -> None:
        contract = subject.threshold_role_contract()
        self.assertEqual(contract["checkpoint_selection_threshold"], 0.5)
        self.assertEqual(contract["global_lambda_selection_threshold"], 0.5)
        self.assertEqual(contract["main_table_threshold"], 0.5)
        self.assertIs(contract["descriptive_sweep_only"], True)
        self.assertIs(contract["sweep_reselects_checkpoint"], False)
        self.assertIs(contract["sweep_reselects_global_lambda"], False)

    def test_probability_evaluation_keeps_sweep_out_of_selection(self) -> None:
        probability = np.asarray([[0.9, 0.0], [0.0, 0.0]], dtype=np.float32)
        target = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        result = subject.evaluate_probability_arrays(
            [probability],
            [target],
            [0.1],
            sweep_thresholds=[0.5, 1.0],
            fa_budgets=[0.0],
        )
        fixed = result["fixed_threshold_0_5"]
        self.assertEqual(fixed["threshold"], 0.5)
        self.assertEqual(fixed["pd"], 1.0)
        self.assertEqual(fixed["unmatched_predicted_pixels"], 0)
        budget = result["descriptive_pd_fa"][
            "best_points_under_fa_budget"
        ]["0"]
        self.assertIs(budget["registered_grid_nonempty_feasible"], True)
        self.assertIs(budget["selected_point_is_empty"], False)
        self.assertEqual(
            result["descriptive_pd_fa"]["selection_effect"], "none"
        )


class FormalIdentityTest(unittest.TestCase):
    def test_dataset_matrix_contains_exactly_three_datasets(self) -> None:
        self.assertEqual(
            tuple(subject.data_protocol.DATASETS),
            ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"),
        )
        self.assertNotIn("SIRST3", subject.data_protocol.DATASETS)
        self.assertEqual(
            subject.NORMALIZATION,
            subject.data_protocol.LEGACY_NORMALIZATION,
        )
        self.assertIsNot(
            subject.NORMALIZATION["NUAA-SIRST"],
            subject.data_protocol.LEGACY_NORMALIZATION["NUAA-SIRST"],
        )

    def test_request_rejects_sirst3_and_enforces_final_lambda(self) -> None:
        with self.assertRaises(ValueError):
            subject.EvaluationRequest(
                "SIRST3", "original", "best_miou"
            ).validate()
        with self.assertRaisesRegex(ValueError, "Final TSS weight"):
            subject.EvaluationRequest(
                "NUAA-SIRST", "final", "best_miou"
            ).validate()
        subject.EvaluationRequest(
            "NUAA-SIRST", "final", "best_miou", 0.005
        ).validate()

    def test_cli_choices_reject_sirst3(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            subject.parse_args(
                [
                    "--dataset",
                    "SIRST3",
                    "--method",
                    "original",
                    "--checkpoint-role",
                    "best_miou",
                    "--run-dir",
                    "/tmp/not-used",
                ]
            )
        self.assertIn("invalid choice", stderr.getvalue())

    def test_run_protocol_must_bind_new_manifest_and_half_threshold(self) -> None:
        request = subject.EvaluationRequest(
            "NUAA-SIRST", "original", "best_miou"
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            manifest = {
                "dataset_order": list(subject.data_protocol.DATASETS),
            }
            summary = {
                "schema": subject.TRAINING_RUN_SCHEMA,
                "dataset": request.dataset,
                "method": request.method,
                "seed": 42,
                "epochs": 1000,
            }
            run_protocol = {
                "schema": subject.TRAINING_RUN_SCHEMA,
                "dataset": request.dataset,
                "method": request.method,
                "training_seed": 42,
                "epochs": 1000,
                "begin_test": 10,
                "eval_every": 10,
                "smoke": False,
                "dataset_counts": {
                    "train": subject.data_protocol.EXPECTED_SPLITS[
                        request.dataset
                    ]["train"]["count"],
                    "test": subject.data_protocol.EXPECTED_SPLITS[
                        request.dataset
                    ]["test"]["count"],
                },
                "test_selected": True,
                "selection_is_optimistic": True,
                "checkpoint_roles": ["best_miou", "best_pd"],
                "metrics": {"threshold": 0.5},
                subject.RUN_DATA_PROTOCOL_FIELD: {
                    "module": "experiments.three_dataset_v2_protocol",
                    "schema": subject.data_protocol.SCHEMA,
                    "manifest_id": subject.data_protocol.MANIFEST_ID,
                    "manifest_sha256": subject._file_sha256(manifest_path),
                    "datasets": list(subject.data_protocol.DATASETS),
                    "sirst3_in_formal_matrix": False,
                },
                "model": {
                    "legacy_model_builder_supported_datasets": [
                        "SIRST3",
                        "NUAA-SIRST",
                        "NUDT-SIRST",
                        "IRSTD-1K",
                    ]
                },
            }
            subject._validate_run_identity(
                request,
                summary,
                run_protocol,
                manifest_path=manifest_path,
                manifest=manifest,
            )
            truncated_protocol = json.loads(json.dumps(run_protocol))
            truncated_protocol["dataset_counts"]["train"] -= 1
            with self.assertRaisesRegex(
                ValueError, "dataset_counts"
            ):
                subject._validate_run_identity(
                    request,
                    summary,
                    truncated_protocol,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
            old_protocol = json.loads(json.dumps(run_protocol))
            del old_protocol[subject.RUN_DATA_PROTOCOL_FIELD]
            with self.assertRaisesRegex(
                ValueError, subject.RUN_DATA_PROTOCOL_FIELD
            ):
                subject._validate_run_identity(
                    request,
                    summary,
                    old_protocol,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )


class EvaluatorProvenanceTest(unittest.TestCase):
    def _runtime_sources(self) -> dict[str, dict[str, str]]:
        return {
            key: {
                "path": str(path),
                "sha256": subject._file_sha256(path),
            }
            for key, path in subject._training_runtime_source_paths().items()
        }

    def _write_run(
        self,
        directory: str,
        *,
        corrupt_runtime_key: str | None = None,
        protocol_declared: str | None = None,
        summary_protocol: str | None = None,
        checkpoint_protocol: str | None = None,
        checkpoint_epoch: int = 20,
    ) -> tuple[
        subject.EvaluationRequest,
        Path,
        Path,
        dict[str, object],
    ]:
        request = subject.EvaluationRequest(
            "NUAA-SIRST", "original", "best_miou"
        )
        root = Path(directory)
        run_dir = root / "run"
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        manifest_path = root / "manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest: dict[str, object] = {
            "dataset_order": list(subject.data_protocol.DATASETS),
        }
        runtime_sources = self._runtime_sources()
        if corrupt_runtime_key is not None:
            runtime_sources[corrupt_runtime_key]["sha256"] = "f" * 64
        protocol: dict[str, object] = {
            "schema": subject.TRAINING_RUN_SCHEMA,
            "dataset": request.dataset,
            "method": request.method,
            "training_seed": 42,
            "epochs": 1000,
            "begin_test": 10,
            "eval_every": 10,
            "smoke": False,
            "dataset_counts": {
                "train": subject.data_protocol.EXPECTED_SPLITS[
                    request.dataset
                ]["train"]["count"],
                "test": subject.data_protocol.EXPECTED_SPLITS[
                    request.dataset
                ]["test"]["count"],
            },
            "test_selected": True,
            "selection_is_optimistic": True,
            "checkpoint_roles": ["best_miou", "best_pd"],
            "metrics": {"threshold": 0.5},
            subject.RUN_DATA_PROTOCOL_FIELD: {
                "module": "experiments.three_dataset_v2_protocol",
                "schema": subject.data_protocol.SCHEMA,
                "manifest_id": subject.data_protocol.MANIFEST_ID,
                "manifest_sha256": subject._file_sha256(manifest_path),
                "datasets": list(subject.data_protocol.DATASETS),
                "sirst3_in_formal_matrix": False,
            },
            "runtime_sources": runtime_sources,
        }
        computed_protocol_sha256 = subject._canonical_sha256(protocol)
        declared = protocol_declared or computed_protocol_sha256
        protocol["protocol_sha256"] = declared
        (run_dir / "protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checkpoint = {
            "schema": subject.TRAINING_RUN_SCHEMA,
            "dataset": request.dataset,
            "method": request.method,
            "seed": 42,
            "checkpoint_role": request.checkpoint_role,
            "selection_source": f"test_{request.dataset}",
            "test_selected": True,
            "selection_is_optimistic": True,
            "epoch": checkpoint_epoch,
            "protocol_sha256": checkpoint_protocol or declared,
            "state_dict": {"weight": torch.tensor([1.0])},
        }
        checkpoint_path = checkpoint_dir / "best_miou.pth.tar"
        torch.save(checkpoint, checkpoint_path)
        summary = {
            "schema": subject.TRAINING_RUN_SCHEMA,
            "status": "complete",
            "dataset": request.dataset,
            "method": request.method,
            "seed": 42,
            "epochs": 1000,
            "protocol_sha256": summary_protocol or declared,
            "best_miou": {"epoch": checkpoint_epoch},
            "checkpoints": {
                "best_miou": {
                    "sha256": subject._file_sha256(checkpoint_path),
                }
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return request, run_dir, manifest_path, manifest

    def test_emitted_source_map_is_complete_sorted_and_real(self) -> None:
        observed = subject.evaluator_source_sha256()
        self.assertEqual(list(observed), sorted(observed))
        required = {
            "experiments/evaluate_three_dataset_v2.py",
            "experiments/three_dataset_v2_protocol.py",
            "experiments/four_dataset_evaluation_protocol_v1.py",
            "experiments/evaluate_pd_fa_sweep.py",
            "experiments/evaluate_tpd_clean_v6_pd_fa.py",
            "experiments/train_tpd_pilot.py",
            "experiments/four_dataset_models_seed42_v1.py",
        }
        self.assertTrue(required.issubset(observed))
        self.assertEqual(
            {key for key in observed if key.startswith("model/")},
            set(subject._model_source_paths()),
        )
        for relative, digest in observed.items():
            self.assertEqual(
                digest,
                subject._file_sha256(subject.REPO_ROOT / relative),
            )

    def test_load_checkpoint_accepts_complete_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, run_dir, manifest_path, manifest = self._write_run(
                directory
            )
            payload, binding = subject.load_checkpoint(
                request,
                run_dir,
                manifest_path=manifest_path,
                manifest=manifest,
            )
        self.assertEqual(payload["checkpoint_role"], "best_miou")
        self.assertEqual(
            binding["protocol"]["payload_sha256"],
            payload["protocol_sha256"],
        )
        self.assertIs(
            binding["training_runtime_sources"]["validated"], True
        )

    def test_protocol_checkpoint_and_summary_hash_mismatches_reject(self) -> None:
        cases = (
            (
                {"protocol_declared": "0" * 64},
                "canonical payload hash",
            ),
            (
                {"summary_protocol": "1" * 64},
                "summary protocol_sha256",
            ),
            (
                {"checkpoint_protocol": "2" * 64},
                "checkpoint protocol_sha256",
            ),
        )
        for options, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                request, run_dir, manifest_path, manifest = self._write_run(
                    directory, **options
                )
                with self.assertRaisesRegex(ValueError, message):
                    subject.load_checkpoint(
                        request,
                        run_dir,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )

    def test_changed_builder_or_model_source_rejects(self) -> None:
        runtime_paths = subject._training_runtime_source_paths()
        architecture_key = next(
            key for key in runtime_paths if key.startswith("architecture::")
        )
        for key in ("model_builder", architecture_key):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                request, run_dir, manifest_path, manifest = self._write_run(
                    directory,
                    corrupt_runtime_key=key,
                )
                with self.assertRaisesRegex(
                    ValueError, "runtime source SHA differs"
                ):
                    subject.load_checkpoint(
                        request,
                        run_dir,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )

    def test_checkpoint_epoch_and_required_metric_contract_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request, run_dir, manifest_path, manifest = self._write_run(
                directory,
                checkpoint_epoch=11,
            )
            with self.assertRaisesRegex(ValueError, "candidate epoch"):
                subject.load_checkpoint(
                    request,
                    run_dir,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
        with self.assertRaisesRegex(ValueError, "required fields"):
            subject._checkpoint_metric_audit(
                {"test_metrics": {}},
                {},
            )


if __name__ == "__main__":
    unittest.main()
