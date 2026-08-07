from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments import evaluate_two_dataset_pbdr_v3_stage1_v1 as evaluator
from experiments import train_two_dataset_pbdr_v3_stage1_v1 as trainer


def _metrics(
    *,
    matched: int = 10,
    total: int = 10,
    fa: float = 0.02,
    miou: float = 0.60,
    niou: float = 0.59,
    tiny_matched: int = 2,
    tiny_total: int = 2,
    loss: float = 0.10,
    valid_pixels: int = 8,
) -> dict[str, float | int]:
    return {
        "threshold": 0.5,
        "matched_target_count": matched,
        "target_count": total,
        "pd": matched / total if total else 0.0,
        "fa": fa,
        "miou": miou,
        "niou": niou,
        "matched_tiny_target_count": tiny_matched,
        "tiny_target_count": tiny_total,
        "tiny_pd": tiny_matched / tiny_total if tiny_total else 0.0,
        "test_loss": loss,
        "valid_pixel_count": valid_pixels,
        "predicted_object_count": 12,
        "unmatched_predicted_object_count": 2,
        "false_objects_per_image": 1.0,
        "pixel_precision": 0.7,
        "pixel_recall": 0.8,
        "pixel_f1": 0.7466666666666666,
    }


def _decision(role: str) -> dict[str, object]:
    current = _metrics()
    candidate = _metrics(miou=0.61)
    decision = evaluator.zero_gate.certify(
        role,
        evaluator.zero_gate.CertificationMetrics.from_mapping(current),
        evaluator.zero_gate.CertificationMetrics.from_mapping(candidate),
    )
    return {
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": current,
        "candidate": candidate,
        "scope": "frozen_internal_validation_split",
        "role": role,
        "decisive_index": decision.decisive_index,
        "decisive_term": decision.decisive_term,
        "minimum_gain": 0.0,
    }


def _checkpoint_record(
    root: Path,
    dataset_name: str,
    role: str,
    family: str,
) -> dict[str, object]:
    path = root / f"{family}-{role}.pth.tar"
    path.write_bytes(f"{family}-{role}".encode("utf-8"))
    return {
        "dataset": dataset_name,
        "checkpoint_role": role,
        "path": str(path.resolve()),
        "sha256": evaluator.models.file_sha256(path),
        "bytes": path.stat().st_size,
        "epoch": 5,
        "state_key_count": 1,
        "state_sha256": ("a" if family == "current" else "b") * 64,
        "schema": f"fake-{family}/v1",
        "protocol_sha256": ("c" if family == "current" else "d") * 64,
    }


def _minimal_validated_dataset(
    root: Path,
    dataset_name: str = "NUDT-SIRST",
) -> evaluator.ValidatedDatasetRuns:
    formal_root = root / "formal"
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_binding = {
        "path": str(manifest.resolve()),
        "sha256": evaluator.models.file_sha256(manifest),
    }
    runs: dict[str, evaluator.ValidatedRun] = {}
    originals: dict[str, dict[str, object]] = {}
    for role in evaluator.PARENT_ROLES:
        run_dir = formal_root / role / "core"
        run_dir.mkdir(parents=True)
        summary = run_dir / "summary.json"
        protocol = run_dir / "protocol.json"
        split = run_dir / "split_manifest.json"
        candidate = run_dir / "selected_candidate.pth.tar"
        summary.write_text("{}\n", encoding="utf-8")
        protocol.write_text("{}\n", encoding="utf-8")
        split.write_text("{}\n", encoding="utf-8")
        candidate.write_bytes(f"candidate-{role}".encode("utf-8"))
        current = _checkpoint_record(root, dataset_name, role, "current")
        original = _checkpoint_record(root, dataset_name, role, "original")
        original.update(
            {
                "fixed_threshold_0_5_metrics": _metrics(),
                "selection_policy": {
                    "threshold": 0.5,
                    "test_selected": True,
                    "selection_is_optimistic": True,
                },
            }
        )
        originals[role] = original
        runs[role] = evaluator.ValidatedRun(
            run_dir=run_dir,
            summary_path=summary,
            protocol_path=protocol,
            split_path=split,
            candidate_path=candidate,
            summary={},
            protocol={"data_root": str(root.resolve())},
            split_manifest={
                "split_sha256": "e" * 64,
                "data_protocol_manifest": manifest_binding,
            },
            candidate={"epoch": 5},
            candidate_state={},
            candidate_sha256=evaluator.models.file_sha256(candidate),
            candidate_state_sha256="f" * 64,
            protocol_sha256="1" * 64,
            dataset_name=dataset_name,
            parent_role=role,
            selected_threshold=0.5,
            internal_decision=_decision(role),
            parent_checkpoint=current,
            runtime_sources={},
        )
    return evaluator.ValidatedDatasetRuns(
        dataset_name=dataset_name,
        formal_root=formal_root,
        runs=runs,
        original_checkpoints=originals,
        original_states={role: {} for role in evaluator.PARENT_ROLES},
        original_authority={
            "authority_manifest": {"path": "fake", "sha256": "2" * 64},
            "selection_policy": {
                "threshold": 0.5,
                "test_selected": True,
                "selection_is_optimistic": True,
            },
        },
        shared_split_sha256="e" * 64,
        shared_runtime_sources_sha256=evaluator.models.canonical_sha256({}),
    )


class TestDynamicSplitReplay(unittest.TestCase):
    def test_split_lengths_are_replayed_not_hard_coded_to_nuaa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            identifiers = [f"train-{index:02d}" for index in range(10)]
            stats = [
                trainer.engine.MaskStats(
                    identifier=identifier,
                    height=32,
                    width=32,
                    target_count=1,
                    target_pixels=16,
                    tiny_target_count=0,
                    minimum_target_area=16,
                    stratum="small_non_tiny",
                )
                for identifier in identifiers
            ]
            development, validation = trainer.engine.stratified_split(
                stats,
                trainer.engine.VAL_FRACTION,
                trainer.engine.SPLIT_SEED,
            )
            self.assertNotEqual((len(development), len(validation)), (170, 43))
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            unsigned = {
                "schema": "sctransnet_two_dataset_pbdr_v3_internal_split_v1/v1",
                "dataset": "NUDT-SIRST",
                "source_split": "official_train_only",
                "official_test_index_opened": False,
                "split_seed": trainer.engine.SPLIT_SEED,
                "val_fraction": trainer.engine.VAL_FRACTION,
                "official_train_ids": identifiers,
                "development_train_ids": development,
                "internal_validation_ids": validation,
                "mask_stats": [asdict(item) for item in stats],
                "official_train_index_sha256": "3" * 64,
                "data_protocol_manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": evaluator.models.file_sha256(manifest),
                },
            }
            split_sha = evaluator.models.canonical_sha256(unsigned)
            (run_dir / "split_manifest.json").write_text(
                json.dumps(dict(unsigned, split_sha256=split_sha)) + "\n",
                encoding="utf-8",
            )
            expected = copy.deepcopy(evaluator.data_protocol.EXPECTED_SPLITS)
            expected["NUDT-SIRST"]["train"] = {
                "count": len(identifiers),
                "file_sha256": "3" * 64,
                "ordered_ids_sha256": evaluator.data_protocol.ordered_ids_sha256(
                    identifiers
                ),
            }
            with mock.patch.object(
                evaluator.data_protocol,
                "EXPECTED_SPLITS",
                expected,
            ):
                _, replayed = evaluator._validate_split_manifest(
                    "NUDT-SIRST",
                    run_dir,
                    {"split_manifest": split_sha},
                )
            self.assertEqual(replayed["development_train_ids"], development)
            self.assertEqual(replayed["internal_validation_ids"], validation)


class TestCertificationMetricProjection(unittest.TestCase):
    def test_full_validation_metrics_equal_gate_projection(self) -> None:
        full = _metrics()
        projection = {
            name: full[name]
            for name in (
                "matched_target_count",
                "target_count",
                "fa",
                "miou",
                "niou",
                "matched_tiny_target_count",
                "tiny_target_count",
                "tiny_pd",
                "test_loss",
            )
        }
        self.assertTrue(
            evaluator._certification_metrics_equal(full, projection)
        )
        forged = dict(projection, miou=float(projection["miou"]) + 0.01)
        self.assertFalse(evaluator._certification_metrics_equal(full, forged))


class _CandidateModel(torch.nn.Module):
    def __init__(self, routed_logit: float, base_logit: float) -> None:
        super().__init__()
        self.routed_logit = routed_logit
        self.base_logit = base_logit
        self.calls = 0

    def forward_for_pbdr_v3_training(self, images: torch.Tensor):
        self.calls += 1
        shape = (images.shape[0], 1, images.shape[-2], images.shape[-1])
        routed = torch.full(shape, self.routed_logit, dtype=torch.float32)
        base = torch.full(shape, self.base_logit, dtype=torch.float32)
        return (), SimpleNamespace(routed_logits=routed, base_logits=base)


class _OriginalModel(torch.nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full(
            (images.shape[0], 1, images.shape[-2], images.shape[-1]),
            self.probability,
            dtype=torch.float32,
        )


class _CountingLoader:
    def __init__(self, count: int) -> None:
        self.count = count
        self.iterations = 0
        self.dataset = [None] * count

    def __iter__(self):
        self.iterations += 1
        for index in range(self.count):
            yield (
                torch.zeros(1, 1, 2, 2, dtype=torch.float32),
                torch.zeros(1, 1, 2, 2, dtype=torch.float32),
                (2, 2),
                [f"sample-{index}"],
            )


class TestOnePassSixModelCollection(unittest.TestCase):
    def test_one_loader_iteration_collects_two_candidate_two_bypass_two_original(self) -> None:
        candidates = {
            "best_miou": _CandidateModel(2.0, -2.0),
            "best_pd": _CandidateModel(1.0, -1.0),
        }
        originals = {
            "best_miou": _OriginalModel(0.25),
            "best_pd": _OriginalModel(0.75),
        }
        loader = _CountingLoader(3)
        cache = evaluator._collect_six_models_one_pass(
            candidates,
            originals,
            loader,  # type: ignore[arg-type]
            torch.device("cpu"),
        )
        self.assertEqual(loader.iterations, 1)
        for role in evaluator.PARENT_ROLES:
            self.assertEqual(candidates[role].calls, 3)
            self.assertEqual(originals[role].calls, 3)
            self.assertEqual(cache["forward_counts"]["current"][role], 3)
        expected = float(torch.sigmoid(torch.tensor(-2.0)))
        np.testing.assert_allclose(
            cache["probabilities"]["current"]["best_miou"][0],
            expected,
        )


class TestZeroMarginOriginalDecision(unittest.TestCase):
    def test_any_strict_best_miou_gain_selects_candidate(self) -> None:
        original = _metrics(miou=0.60)
        candidate = _metrics(miou=math.nextafter(0.60, math.inf))
        decision = evaluator._official_role_decision(
            "best_miou", candidate, original
        )
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(decision["minimum_gain"], 0.0)
        self.assertEqual(decision["decisive_term"], "higher_miou")

    def test_exact_tie_retains_original(self) -> None:
        metrics = _metrics()
        decision = evaluator._official_role_decision(
            "best_miou", metrics, metrics
        )
        self.assertEqual(decision["selected"], "original")
        self.assertIsNone(decision["decisive_term"])

    def test_best_pd_lower_pd_loses_despite_better_later_metrics(self) -> None:
        original = _metrics(matched=10, total=10, fa=0.5, miou=0.1)
        candidate = _metrics(matched=9, total=10, fa=0.0, miou=0.99)
        decision = evaluator._official_role_decision(
            "best_pd", candidate, original
        )
        self.assertEqual(decision["selected"], "original")
        self.assertEqual(decision["decisive_term"], "higher_pd")
        self.assertIn("pd", decision["metric_comparison"]["regressed"])

    def test_best_pd_equal_pd_then_any_lower_fa_wins(self) -> None:
        original = _metrics(fa=0.02)
        candidate = _metrics(fa=math.nextafter(0.02, 0.0))
        decision = evaluator._official_role_decision(
            "best_pd", candidate, original
        )
        self.assertEqual(decision["selected"], "candidate")
        self.assertEqual(decision["decisive_term"], "lower_fa")

    def test_tiny_target_denominator_mismatch_fails_closed(self) -> None:
        original = _metrics(tiny_matched=2, tiny_total=4)
        candidate = _metrics(
            miou=0.61,
            tiny_matched=2,
            tiny_total=5,
        )
        with self.assertRaises(evaluator.PBDRV3EvaluationProtocolError):
            evaluator._official_role_decision(
                "best_miou", candidate, original
            )


class TestDatasetGlobalClaimBoundary(unittest.TestCase):
    def test_existing_claim_fails_before_dataset_or_loader_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _minimal_validated_dataset(root)
            manifest = Path(
                validated.runs["best_miou"].split_manifest[
                    "data_protocol_manifest"
                ]["path"]
            )
            claim = validated.formal_root / "official_test_access_claim.json"
            claim.write_text("already claimed\n", encoding="utf-8")
            prepared = evaluator.PreparedModels(
                candidates={},
                originals={},
                candidate_metadata={},
                original_metadata={},
            )
            previous_cudnn = torch.backends.cudnn.allow_tf32
            previous_matmul = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                with mock.patch.object(
                    evaluator,
                    "OfficialTestDataset",
                    side_effect=AssertionError("official dataset was constructed"),
                ) as dataset_constructor:
                    with self.assertRaisesRegex(
                        evaluator.PBDRV3EvaluationProtocolError,
                        "already claimed",
                    ):
                        evaluator._evaluate_official_test(
                            validated,
                            prepared,
                            data_root=root,
                            protocol_manifest=manifest,
                            device=torch.device("cpu"),
                            workers=0,
                            access_claim_path=claim,
                        )
            finally:
                torch.backends.cudnn.allow_tf32 = previous_cudnn
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            dataset_constructor.assert_not_called()

    def test_dataset_evaluation_discloses_original_optimism_and_selects_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validated = _minimal_validated_dataset(root)
            identifiers = ["test-a", "test-b"]

            class TinyDataset(Dataset):
                sample_ids = identifiers

                def __len__(self) -> int:
                    return len(self.sample_ids)

                def __getitem__(self, index: int):
                    return (
                        torch.zeros(1, 2, 2),
                        torch.zeros(1, 2, 2),
                        (2, 2),
                        self.sample_ids[index],
                    )

            cache = {
                "probabilities": {
                    family: {role: [] for role in evaluator.PARENT_ROLES}
                    for family in ("candidate", "current", "original")
                },
                "losses": {
                    family: {role: [] for role in evaluator.PARENT_ROLES}
                    for family in ("candidate", "current", "original")
                },
                "targets": [np.zeros((2, 2)), np.zeros((2, 2))],
                "identifiers": identifiers,
                "loader_iteration_count": 1,
                "forward_counts": {
                    family: {role: 2 for role in evaluator.PARENT_ROLES}
                    for family in ("candidate", "current", "original")
                },
            }
            points = [
                _metrics(miou=0.61),  # candidate best_miou wins
                _metrics(matched=9),  # candidate best_pd loses
                _metrics(),
                _metrics(),
                _metrics(miou=0.60),
                _metrics(matched=10),
            ]

            def metric_points(*args, **kwargs):
                del args, kwargs
                return {"0.50": points.pop(0)}

            expected = copy.deepcopy(evaluator.data_protocol.EXPECTED_SPLITS)
            expected["NUDT-SIRST"]["test"].update(
                {
                    "count": len(identifiers),
                    "ordered_ids_sha256": evaluator.data_protocol.ordered_ids_sha256(
                        identifiers
                    ),
                }
            )
            prepared = evaluator.PreparedModels(
                candidates={},
                originals={},
                candidate_metadata={
                    role: {"inference_state_sha256": "4" * 64}
                    for role in evaluator.PARENT_ROLES
                },
                original_metadata={role: {} for role in evaluator.PARENT_ROLES},
            )
            manifest = Path(
                validated.runs["best_miou"].split_manifest[
                    "data_protocol_manifest"
                ]["path"]
            )
            claim_path = validated.formal_root / "claim.json"
            previous_cudnn = torch.backends.cudnn.allow_tf32
            previous_matmul = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                with (
                    mock.patch.object(
                        evaluator,
                        "OfficialTestDataset",
                        return_value=TinyDataset(),
                    ),
                    mock.patch.object(
                        evaluator,
                        "_collect_six_models_one_pass",
                        return_value=cache,
                    ),
                    mock.patch.object(
                        evaluator,
                        "_metric_points",
                        side_effect=metric_points,
                    ),
                    mock.patch.object(
                        evaluator.data_protocol,
                        "EXPECTED_SPLITS",
                        expected,
                    ),
                ):
                    evaluation, deployments = evaluator._evaluate_official_test(
                        validated,
                        prepared,
                        data_root=root,
                        protocol_manifest=manifest,
                        device=torch.device("cpu"),
                        workers=0,
                        access_claim_path=claim_path,
                    )
            finally:
                torch.backends.cudnn.allow_tf32 = previous_cudnn
                torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            self.assertEqual(
                evaluation["candidate_vs_same_role_original"]["best_miou"][
                    "selected"
                ],
                "candidate",
            )
            self.assertEqual(
                evaluation["candidate_vs_same_role_original"]["best_pd"][
                    "selected"
                ],
                "original",
            )
            self.assertTrue(
                evaluation["original_selection_disclosure"][
                    "selection_is_optimistic"
                ]
            )
            self.assertEqual(deployments["best_miou"]["selected"], "candidate")
            self.assertEqual(deployments["best_pd"]["selected"], "original")
            self.assertEqual(
                evaluation["official_test_loader_construction_count"], 1
            )
            bundle = evaluator._make_publication_bundle(
                "NUDT-SIRST", evaluation, deployments
            )
            committed_evaluation, committed_templates = (
                evaluator._validate_publication_bundle(
                    validated, claim_path, bundle
                )
            )
            evaluation_path = validated.formal_root / "evaluation.json"
            deployment_paths = {
                role: validated.runs[role].run_dir / "deployment.json"
                for role in evaluator.PARENT_ROLES
            }
            first_evaluation, first_deployments = (
                evaluator._materialize_publication_views(
                    evaluation_path,
                    deployment_paths,
                    committed_evaluation,
                    committed_templates,
                )
            )
            first_sha = evaluator.models.file_sha256(first_evaluation)
            deployment_paths["best_pd"].unlink()
            second_evaluation, second_deployments = (
                evaluator._materialize_publication_views(
                    evaluation_path,
                    deployment_paths,
                    committed_evaluation,
                    committed_templates,
                )
            )
            self.assertEqual(first_evaluation, second_evaluation)
            self.assertEqual(
                evaluator.models.file_sha256(second_evaluation), first_sha
            )
            self.assertEqual(set(first_deployments), set(second_deployments))
            claim_payload = json.loads(claim_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(claim_payload["candidate_checkpoint_sha256"]),
                set(evaluator.PARENT_ROLES),
            )


class TestFormalAuthorityPaths(unittest.TestCase):
    def test_claim_publication_run_dirs_and_results_root_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                {"access_claim_output": root / "alternate-claim.json"},
                {"evaluation_output": root / "alternate-evaluation.json"},
                {
                    "deployment_outputs": {
                        role: root / f"alternate-{role}.json"
                        for role in evaluator.PARENT_ROLES
                    }
                },
                {
                    "run_directories": {
                        role: root / role for role in evaluator.PARENT_ROLES
                    }
                },
                {"results_root": root},
            )
            for overrides in cases:
                with self.subTest(overrides=overrides):
                    with self.assertRaises(
                        evaluator.PBDRV3EvaluationProtocolError
                    ):
                        evaluator.run(
                            dataset_name="NUDT-SIRST", **overrides
                        )


if __name__ == "__main__":
    unittest.main()
