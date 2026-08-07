from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import evaluate_three_dataset_pbdr_v4_v1 as evaluator
from experiments import pbdr_v3_residual_calibration as calibration
from experiments import pbdr_v4_candidate_pool as candidate_pool_module


torch.set_num_threads(1)


class FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits.clone())
        self.forward_count = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.forward_count += int(images.shape[0])
        logits = self.fixed_logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="nearest")
        return logits.expand(images.shape[0], -1, -1, -1)


class FakeOriginal(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.outc = FixedHead(logits)
        self.mode = "test"

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.outc(images)
        return torch.sigmoid(raw)


class FakeOriginalMismatch(FakeOriginal):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.outc(images)
        return torch.full_like(raw, 0.99)


class FakeCurrent(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits.clone())
        self.mode = "test"
        self.forward_count = 0

    def forward_for_pbdr_v4_training(
        self, images: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], SimpleNamespace]:
        self.forward_count += int(images.shape[0])
        logits = self.fixed_logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="nearest")
        logits = logits.expand(images.shape[0], -1, -1, -1)
        return (torch.sigmoid(logits),), SimpleNamespace(
            candidate_base_logits=logits,
            routed_logits=logits,
        )


class FakeV3(nn.Module):
    def __init__(self, base: torch.Tensor, routed: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base", base.clone())
        self.register_buffer("routed", routed.clone())
        self.mode = "test"
        self.forward_count = 0

    def forward_for_pbdr_v3_training(
        self, images: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], SimpleNamespace]:
        self.forward_count += int(images.shape[0])
        base = self.base
        routed = self.routed
        if base.shape[-2:] != images.shape[-2:]:
            base = F.interpolate(base, size=images.shape[-2:], mode="nearest")
            routed = F.interpolate(routed, size=images.shape[-2:], mode="nearest")
        base = base.expand(images.shape[0], -1, -1, -1)
        routed = routed.expand(images.shape[0], -1, -1, -1)
        return (torch.sigmoid(routed),), SimpleNamespace(
            base_logits=base,
            routed_logits=routed,
        )


class FakeV4(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits.clone())
        self.mode = "test"
        self.forward_count = 0

    def forward_for_pbdr_v4_training(
        self, images: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], SimpleNamespace]:
        self.forward_count += int(images.shape[0])
        logits = self.fixed_logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="nearest")
        logits = logits.expand(images.shape[0], -1, -1, -1)
        return (torch.sigmoid(logits),), SimpleNamespace(
            candidate_base_logits=logits,
            routed_logits=logits,
        )


class OnePassLoader:
    def __init__(self, batches: list[object]) -> None:
        self.batches = batches
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise AssertionError("official loader was iterated more than once")
        yield from self.batches


class SyntheticEvaluationFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.source_lock = {
            "schema": "synthetic-source-lock",
            "source_lock_sha256": "a" * 64,
            "official_test_accessed": False,
            "sources": {
                relative: {"sha256": "1" * 64}
                for relative in evaluator.REQUIRED_SOURCE_LOCK_PATHS
            },
        }
        self.projection = {
            "schema": evaluator.split_authority.SCHEMA,
            "status": "frozen_v3_split_authority_projection",
            "dataset_order": list(evaluator.DATASETS),
            "official_test_accessed": False,
            "split_reconstruction_performed": False,
            "datasets": {
                dataset: {
                    "dataset": dataset,
                    "official_test_accessed": False,
                }
                for dataset in evaluator.DATASETS
            },
        }
        self.projection["projection_sha256"] = (
            evaluator.split_authority.canonical_sha256(self.projection)
        )
        target_only = torch.full((1, 1, 8, 8), -4.0)
        target_only[0, 0, 1, 1] = 4.0
        target_and_false = target_only.clone()
        target_and_false[0, 0, 6, 6] = 4.0
        missed = torch.full((1, 1, 8, 8), -4.0)
        kinds = (
            "original_checkpoint",
            "current_checkpoint",
            "v3_residual_calibration",
            "v4_stage1_checkpoint",
            "v4_stage2_checkpoint",
        )
        self.pools: dict[str, dict[str, object]] = {}
        self.models: dict[str, dict[str, nn.Module]] = {}
        self.runtimes: dict[str, dict[str, evaluator.CandidateRuntime]] = {}
        for role_index, role in enumerate(evaluator.ROLES):
            artifacts = []
            for family_index, (family, kind) in enumerate(
                zip(evaluator.FAMILY_ORDER, kinds, strict=True)
            ):
                path = self.root / f"artifact_{role}_{family_index}.bin"
                path.write_bytes(f"artifact-{role}-{family}".encode("utf-8"))
                artifacts.append(
                    candidate_pool_module.CandidateArtifact(
                        family=family,
                        name=f"{role}-{family}-frozen",
                        kind=kind,
                        artifact_path=str(path.resolve()),
                        artifact_sha256=candidate_pool_module.file_sha256(path),
                        state_sha256=f"{role_index * 5 + family_index + 2:x}" * 64,
                        configuration_sha256=(
                            f"{role_index * 5 + family_index + 8:x}"[-1] * 64
                        ),
                    )
                )
            pool = candidate_pool_module.build_candidate_pool(
                dataset="NUAA-SIRST",
                role=role,
                source_lock_sha256=self.source_lock["source_lock_sha256"],
                split_projection_sha256=self.projection["projection_sha256"],
                candidates=artifacts,
            )
            self.pools[role] = pool
            role_models: dict[str, nn.Module] = {
                "Original": FakeOriginal(missed),
                "Current": FakeCurrent(target_and_false),
                "V3-calibrated": FakeV3(target_and_false, target_and_false),
                "V4-Stage1": FakeV4(target_and_false),
                "V4-Stage2": FakeV4(target_only),
            }
            for model in role_models.values():
                model.eval()
                model.mode = "test"
            self.models[role] = role_models
            records = {item["family"]: item for item in pool["candidates"]}
            self.runtimes[role] = {
                family: evaluator.CandidateRuntime(
                    family=family,
                    name=records[family]["name"],
                    model=role_models[family],
                    artifact_path=records[family]["artifact_path"],
                    artifact_sha256=records[family]["artifact_sha256"],
                    state_sha256=records[family]["state_sha256"],
                    configuration_sha256=records[family]["configuration_sha256"],
                    calibration=(
                        calibration.PBDR_V3_ANCHOR
                        if family == "V3-calibrated"
                        else None
                    ),
                    checkpoint_binding={
                        "dataset": "NUAA-SIRST",
                        "role": role,
                        "official_test_accessed": False,
                    },
                )
                for family in evaluator.FAMILY_ORDER
            }
        image = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        target[0, 0, 1, 1] = 1.0
        self.batches = [
            (image, target, (torch.tensor([8]), torch.tensor([8])), ["a"]),
            (image, target, (torch.tensor([8]), torch.tensor([8])), ["b"]),
        ]

    def __enter__(self) -> "SyntheticEvaluationFixture":
        return self

    def __exit__(self, *_: object) -> None:
        self.temporary.cleanup()

    def candidate_factory(self, **kwargs: object):
        return self.runtimes[str(kwargs["role"])]

    def validation_patches(self):
        return (
            mock.patch.object(
                evaluator.source_lock_module,
                "validate_source_lock",
                return_value=self.source_lock,
            ),
            mock.patch.object(
                evaluator.split_authority,
                "build_projection",
                return_value=self.projection,
            ),
        )


class ThreeDatasetPBDRV4EvaluatorTests(unittest.TestCase):
    def test_one_claim_one_loader_pass_ten_models_and_zero_margin_winners(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            events: list[str] = []
            loader = OnePassLoader(fixture.batches)

            def factory() -> OnePassLoader:
                self.assertTrue((fixture.run_dir / "official_claim.json").is_file())
                events.append("loader_after_claim")
                return loader

            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch:
                bundle = evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=fixture.candidate_factory,
                    loader_factory=factory,
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )

            self.assertEqual(events, ["loader_after_claim"])
            self.assertEqual(loader.iteration_count, 1)
            self.assertEqual(bundle["loader_iteration_count"], 1)
            self.assertEqual(bundle["sample_count"], 2)
            self.assertEqual(
                bundle["forward_counts"],
                {
                    f"{role}::{family}": 2
                    for role in evaluator.ROLES
                    for family in evaluator.FAMILY_ORDER
                },
            )
            metrics = bundle["metrics"]
            self.assertEqual(set(metrics["all_performance"]), set(evaluator.ROLES))
            for role in evaluator.ROLES:
                self.assertEqual(
                    set(metrics["all_performance"][role]),
                    set(evaluator.FAMILY_ORDER),
                )
                self.assertEqual(
                    metrics["zero_margin_selection"][role]["winner_family"],
                    "V4-Stage2",
                )
            self.assertTrue(metrics["operational_test_selected"])
            self.assertTrue(metrics["selection_is_optimistic"])
            manifest = metrics["metric_manifest"]
            self.assertEqual(manifest["threshold"], 0.5)
            self.assertEqual(manifest["probability_comparison"], "strict_greater_than")
            self.assertEqual(manifest["connectivity"], 2)
            self.assertEqual(manifest["match_radius"], 3.0)
            self.assertEqual(manifest["tiny_area"], 9)

            for role in evaluator.ROLES:
                self.assertEqual(
                    fixture.models[role]["Original"].outc.forward_count, 3
                )
                for family in evaluator.FAMILY_ORDER[1:]:
                    self.assertEqual(
                        fixture.models[role][family].forward_count, 3
                    )
            self.assertFalse((fixture.run_dir / "consumed_failure.json").exists())

    def test_committed_replay_never_rebuilds_models_or_loader(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch:
                first = evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=fixture.candidate_factory,
                    loader_factory=lambda: OnePassLoader(fixture.batches),
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )

            def forbidden(**_: object):
                raise AssertionError("committed replay rebuilt candidates")

            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch:
                replay = evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=forbidden,
                    loader_factory=lambda: (_ for _ in ()).throw(
                        AssertionError("committed replay constructed loader")
                    ),
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )
            self.assertEqual(first, replay)

    def test_preflight_failure_creates_no_claim_and_never_calls_loader(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            loader_called = False

            def loader_factory():
                nonlocal loader_called
                loader_called = True
                return []

            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch, self.assertRaisesRegex(
                evaluator.PBDRV4EvaluationError, "candidate runtime"
            ):
                evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=lambda **_: {},
                    loader_factory=loader_factory,
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )
            self.assertFalse(loader_called)
            self.assertFalse((fixture.run_dir / "official_claim.json").exists())
            self.assertFalse((fixture.run_dir / "consumed_failure.json").exists())

    def test_failure_after_claim_is_consumed_and_cannot_retry(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            attempts = 0

            def broken_loader():
                nonlocal attempts
                attempts += 1
                self.assertTrue((fixture.run_dir / "official_claim.json").is_file())
                raise RuntimeError("synthetic post-claim failure")

            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch, self.assertRaisesRegex(
                RuntimeError, "synthetic post-claim failure"
            ):
                evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=fixture.candidate_factory,
                    loader_factory=broken_loader,
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )
            self.assertEqual(attempts, 1)
            self.assertTrue((fixture.run_dir / "consumed_failure.json").is_file())

            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch, self.assertRaisesRegex(
                evaluator.official_once.PBDRV4OfficialOnceError,
                "already consumed",
            ):
                evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=fixture.candidate_factory,
                    loader_factory=broken_loader,
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )
            self.assertEqual(attempts, 1)

    def test_runtime_cross_role_or_artifact_tamper_fails_before_claim(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            tampered = {
                role: dict(role_runtimes)
                for role, role_runtimes in fixture.runtimes.items()
            }
            original = fixture.runtimes["best_miou"]["Original"]
            tampered["best_miou"]["Original"] = evaluator.CandidateRuntime(
                family=original.family,
                name=original.name,
                model=original.model,
                artifact_path=original.artifact_path,
                artifact_sha256=original.artifact_sha256,
                state_sha256="f" * 64,
                configuration_sha256=original.configuration_sha256,
                calibration=None,
                checkpoint_binding={
                    "dataset": "NUAA-SIRST",
                    "role": "best_pd",
                    "official_test_accessed": False,
                },
            )
            source_patch, split_patch = fixture.validation_patches()
            with source_patch, split_patch, self.assertRaises(
                evaluator.PBDRV4EvaluationError
            ):
                evaluator.evaluate_official_once(
                    run_dir=fixture.run_dir,
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=lambda **kwargs: tampered[str(kwargs["role"])],
                    loader_factory=lambda: OnePassLoader(fixture.batches),
                    device=torch.device("cpu"),
                    expected_gpu_uuid=None,
                    operational_test_selected=True,
                )
            self.assertFalse((fixture.run_dir / "official_claim.json").exists())

    def test_original_hook_and_v3_frozen_calibration_return_raw_logits(self) -> None:
        image = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        raw = torch.tensor([[[[-2.0, -1.0], [1.0, 2.0]]]])
        original_model = FakeOriginal(raw)
        original = evaluator.CandidateRuntime(
            family="Original",
            name="original",
            model=original_model,
            artifact_path="/tmp/original",
            artifact_sha256="1" * 64,
            state_sha256="2" * 64,
            configuration_sha256="3" * 64,
            calibration=None,
            checkpoint_binding={
                "dataset": "NUAA-SIRST",
                "role": "best_miou",
                "official_test_accessed": False,
            },
        )
        self.assertTrue(torch.equal(evaluator.forward_candidate_logits(original, image), raw))

        mismatched_original = evaluator.CandidateRuntime(
            family="Original",
            name="mismatched-original",
            model=FakeOriginalMismatch(raw),
            artifact_path="/tmp/mismatched-original",
            artifact_sha256="1" * 64,
            state_sha256="2" * 64,
            configuration_sha256="3" * 64,
            calibration=None,
            checkpoint_binding={
                "dataset": "NUAA-SIRST",
                "role": "best_miou",
                "official_test_accessed": False,
            },
        )
        with self.assertRaisesRegex(
            evaluator.PBDRV4EvaluationError,
            "probability.*raw logits",
        ):
            evaluator.forward_candidate_logits(mismatched_original, image)

        base = torch.zeros_like(raw)
        routed = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]]])
        config = calibration.ResidualCalibration(2.0, 0.5, 0.1)
        v3 = evaluator.CandidateRuntime(
            family="V3-calibrated",
            name="v3",
            model=FakeV3(base, routed),
            artifact_path="/tmp/v3",
            artifact_sha256="4" * 64,
            state_sha256="5" * 64,
            configuration_sha256="6" * 64,
            calibration=config,
            checkpoint_binding={
                "dataset": "NUAA-SIRST",
                "role": "best_miou",
                "official_test_accessed": False,
            },
        )
        expected = calibration.apply_residual_calibration(base, routed - base, config)
        self.assertTrue(torch.equal(evaluator.forward_candidate_logits(v3, image), expected))

    def test_preclaim_runs_seed42_then_ten_maximum_size_forwards(self) -> None:
        with SyntheticEvaluationFixture() as fixture:
            events: list[str] = []
            _, split_patch = fixture.validation_patches()

            def configure(seed: int = 42) -> dict[str, object]:
                events.append(f"determinism:{seed}")
                return {"seed": seed, "deterministic_algorithms": True}

            def validate(*_args: object, **_kwargs: object) -> dict[str, object]:
                events.append("source_lock")
                return fixture.source_lock

            with split_patch, mock.patch.object(
                evaluator.training_core,
                "configure_determinism",
                side_effect=configure,
            ), mock.patch.object(
                evaluator.source_lock_module,
                "validate_source_lock",
                side_effect=validate,
            ):
                prepared = evaluator.prepare_evaluation(
                    dataset="NUAA-SIRST",
                    source_lock=fixture.source_lock,
                    split_projection=fixture.projection,
                    candidate_pools=fixture.pools,
                    candidate_factory=fixture.candidate_factory,
                    device=torch.device("cpu"),
                )

            self.assertEqual(events[0], "determinism:42")
            self.assertEqual(events.count("source_lock"), 2)
            self.assertEqual(prepared.audit["synthetic_preflight_size"], [512, 512])
            self.assertEqual(
                prepared.audit["synthetic_preflight_forward_counts"],
                {
                    f"{role}::{family}": 1
                    for role in evaluator.ROLES
                    for family in evaluator.FAMILY_ORDER
                },
            )
            self.assertFalse(prepared.audit["dataset_or_loader_constructed_before_claim"])

    def test_v4_candidate_envelope_rejects_smoke_test_access_and_margin(self) -> None:
        payload: dict[str, object] = {
            "schema": "candidate",
            "smoke": False,
            "official_test_accessed": False,
            "official_test_data_accessed": False,
            "performance_acceptance_margin": None,
            "state_dict": {"weight": torch.tensor([1.0])},
        }
        payload["candidate_manifest_sha256"] = (
            evaluator.v4_candidate_manifest_sha256(payload)
        )
        evaluator.validate_v4_candidate_envelope(payload)
        for field, invalid in (
            ("smoke", True),
            ("official_test_accessed", True),
            ("official_test_data_accessed", True),
            ("performance_acceptance_margin", 0.001),
        ):
            tampered = dict(payload)
            tampered[field] = invalid
            tampered["candidate_manifest_sha256"] = (
                evaluator.v4_candidate_manifest_sha256(tampered)
            )
            with self.subTest(field=field), self.assertRaises(
                evaluator.PBDRV4EvaluationError
            ):
                evaluator.validate_v4_candidate_envelope(tampered)

        manifest_tampered = dict(payload)
        manifest_tampered["schema"] = "tampered"
        with self.assertRaisesRegex(
            evaluator.PBDRV4EvaluationError,
            "candidate manifest",
        ):
            evaluator.validate_v4_candidate_envelope(manifest_tampered)

    def test_v3_candidate_envelope_replays_scope_state_and_provenance(self) -> None:
        state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
        state_sha = evaluator.v4_models.state_semantic_sha256(state)
        config = calibration.PBDR_V3_ANCHOR
        pool = {
            "source_lock_sha256": "a" * 64,
            "split_projection_sha256": "b" * 64,
        }
        payload: dict[str, object] = {
            "schema": evaluator.V3_CALIBRATED_ARTIFACT_SCHEMA,
            "family": "V3-calibrated",
            "dataset": "NUDT-SIRST",
            "role": "best_pd",
            "state_dict": state,
            "state_key_count": 1,
            "state_sha256": state_sha,
            "state_semantic_sha256": state_sha,
            "calibration": config.as_dict(),
            "configuration_sha256": evaluator.candidate_configuration_sha256(
                family="V3-calibrated",
                dataset="NUDT-SIRST",
                role="best_pd",
                details={
                    "state_sha256": state_sha,
                    "calibration": config.as_dict(),
                    "selected_on": "internal_validation",
                },
            ),
            "selected_on": "internal_validation",
            "v3_candidate_binding": {"path": "/tmp/v3"},
            "sweep_binding": {"path": "/tmp/sweep"},
            "cache_binding": {"path": "/tmp/cache"},
            "source_lock_sha256": pool["source_lock_sha256"],
            "split_projection_sha256": pool["split_projection_sha256"],
            "fixed_probability_rule": "strict_greater_than_0.5",
            "performance_acceptance_margin": None,
            "official_test_accessed": False,
        }
        payload["artifact_manifest_sha256"] = (
            evaluator.v3_artifact_manifest_sha256(payload)
        )
        evaluator.validate_v3_candidate_envelope(
            payload,
            dataset="NUDT-SIRST",
            role="best_pd",
            pool=pool,
        )
        cases = (
            ("fixed_probability_rule", "greater_than_or_equal_0.5"),
            ("performance_acceptance_margin", 0.001),
            ("source_lock_sha256", "c" * 64),
            ("split_projection_sha256", "d" * 64),
            ("state_key_count", 2),
            ("sweep_binding", []),
            ("cache_binding", []),
            ("v3_candidate_binding", []),
        )
        for field, invalid in cases:
            tampered = dict(payload)
            tampered[field] = invalid
            tampered["artifact_manifest_sha256"] = (
                evaluator.v3_artifact_manifest_sha256(tampered)
            )
            with self.subTest(field=field), self.assertRaises(
                evaluator.PBDRV4EvaluationError
            ):
                evaluator.validate_v3_candidate_envelope(
                    tampered,
                    dataset="NUDT-SIRST",
                    role="best_pd",
                    pool=pool,
                )

        manifest_tampered = dict(payload)
        manifest_tampered["selected_on"] = "official_test"
        with self.assertRaises(evaluator.PBDRV4EvaluationError):
            evaluator.validate_v3_candidate_envelope(
                manifest_tampered,
                dataset="NUDT-SIRST",
                role="best_pd",
                pool=pool,
            )

    def test_formal_gpu_uuid_is_dataset_fixed_and_cli_required(self) -> None:
        expected = evaluator.FORMAL_GPU_UUIDS["NUDT-SIRST"]
        properties = SimpleNamespace(uuid=expected.removeprefix("GPU-"))
        with mock.patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=properties,
        ):
            observed = evaluator.validate_evaluation_device(
                dataset="NUDT-SIRST",
                device=torch.device("cuda:0"),
                expected_gpu_uuid=expected,
            )
        self.assertEqual(observed, expected)
        with self.assertRaisesRegex(evaluator.PBDRV4EvaluationError, "fixed UUID"):
            evaluator.validate_evaluation_device(
                dataset="NUDT-SIRST",
                device=torch.device("cuda:0"),
                expected_gpu_uuid="GPU-forbidden-physical-gpu2",
            )
        with self.assertRaises(SystemExit):
            evaluator.parse_args(
                [
                    "--dataset",
                    "NUDT-SIRST",
                    "--run-dir",
                    "/tmp/run",
                    "--source-lock",
                    "/tmp/source",
                    "--split-projection",
                    "/tmp/split",
                    "--best-miou-candidate-pool",
                    "/tmp/miou",
                    "--best-pd-candidate-pool",
                    "/tmp/pd",
                ]
            )


if __name__ == "__main__":
    unittest.main()
