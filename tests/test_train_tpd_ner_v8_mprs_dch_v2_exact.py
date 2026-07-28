from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact_resume
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_ner_v8_mprs_dch_exact as v1_exact
from experiments import train_tpd_ner_v8_mprs_dch_v2_exact as entry


V2_ON = entry.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON


class StateScaler:
    def __init__(self, scale: float = 128.0, updates: int = 0) -> None:
        self.scale = float(scale)
        self.updates = int(updates)

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale, "updates": self.updates}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self.updates = int(state["updates"])


class TinyTrajectoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(value).flatten(1))


@contextmanager
def cpu_rng_environment():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
        mock.patch.object(torch.cuda, "get_rng_state_all"),
        mock.patch.object(torch.cuda, "set_rng_state_all"),
    ):
        yield


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse(trailing: list[str]) -> argparse.Namespace:
    return entry.parse_args(
        [
            "--variant",
            V2_ON,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            *trailing,
        ]
    )


def make_components(seed: int) -> dict[str, object]:
    seed_all(seed)
    model = TinyTrajectoryModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=entry.FORMAL_BASE_LR)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(entry.TRAINING_SEED)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": generator,
    }


def tiny_metadata() -> dict[str, object]:
    manifest = {
        "schema": entry.ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": V2_ON,
        "model": "tests.TinyTrajectoryModel",
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_version": "v2_rms_centered_arctangent",
        "relay_width": entry.RELAY_WIDTH,
        "relay_rms_eps": 1e-6,
        "gate_bias": False,
        "eps": entry.FORMAL_EPS,
    }
    return {
        "variant": V2_ON,
        "candidate_family": "fixture-v2-ner",
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_version": "v2_rms_centered_arctangent",
        "relay_width": entry.RELAY_WIDTH,
        "relay_initialization_seed": entry.RELAY_INITIALIZATION_SEED,
        "required_control": entry.V1_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "architecture_manifest": manifest,
        "architecture_id": entry.canonical_sha256(manifest),
    }


def selection_policy() -> exact_runner.SelectionPolicy:
    return exact_runner.pd_miou_selection_policy(
        stored_metrics=entry.STORED_VALIDATION_METRICS
    )


def make_spec(
    args: argparse.Namespace,
    components: dict[str, object],
    *,
    initial_model_state_sha256: str | None = None,
    initial_rng: dict | None = None,
    policy: dict | None = None,
) -> exact_runner.ExactRunSpec:
    model = components["model"]
    optimizer = components["optimizer"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is invalid")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer is invalid")
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=tiny_metadata(),
        optimizer=optimizer,
        scaler=components["scaler"],
        initialization_contract=exact_runner.fresh_initialization_contract(),
        initial_model_state_sha256=(
            initial_model_state_sha256
            or exact_runner.initial_model_state_sha256(model)
        ),
        initial_rng=(
            copy.deepcopy(initial_rng)
            if initial_rng is not None
            else exact_runner.initial_rng_contract()
        ),
        selection_policy=(
            copy.deepcopy(policy)
            if policy is not None
            else selection_policy().normalized()
        ),
        source_locks={entry.SOURCE_LOCK_KEY: "1" * 64},
        split_records={
            "train": exact_runner.OrderedFingerprint.from_values(
                "train", ("a", "b")
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation", ("c",)
            ),
        },
        data_records={
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples", ("a:image", "b:image")
            ),
            "normalization": exact_runner.OrderedFingerprint.from_values(
                "normalization", ('{"mean":1.0,"std":2.0}',)
            ),
        },
        environment={"name": "cpu-v2-fixture"},
    )


def make_runner(
    directory: Path,
    args: argparse.Namespace,
    components: dict[str, object],
    *,
    spec: exact_runner.ExactRunSpec | None = None,
) -> entry.TPDNERV8V2ExactRunner:
    model = components["model"]
    optimizer = components["optimizer"]
    generator = components["loader_generator"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is invalid")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer is invalid")
    if not isinstance(generator, torch.Generator):
        raise TypeError("fixture generator is invalid")
    return entry.TPDNERV8V2ExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=components["scaler"],
        loader_generator=generator,
        spec=spec or make_spec(args, components),
        selection_policy=selection_policy(),
        compatibility_payload_factory=entry.EvaluatorCheckpointAdapter(
            model_metadata=tiny_metadata(),
            split_hashes={"train": "a" * 64},
        ),
    )


def validation_metrics(epoch: int, loss: float) -> dict[str, int | float]:
    return {
        "val_loss": loss,
        "miou": 0.50 + 0.02 * epoch,
        "niou": 0.48 + 0.02 * epoch,
        "pixel_precision": 0.80 + 0.01 * epoch,
        "pixel_recall": 0.75 + 0.01 * epoch,
        "pixel_f1": 0.77 + 0.01 * epoch,
        "pd": 0.70 + 0.01 * epoch,
        "tiny_pd": 0.60 + 0.01 * epoch,
        "fa": 1.0e-5 / epoch,
        "false_objects_per_image": 0.1 / epoch,
        "target_count": 189,
        "matched_target_count": 180 + epoch,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 37 + min(epoch, 2),
        "predicted_object_count": 200 - epoch,
        "unmatched_predicted_object_count": 10 - min(epoch, 9),
        "valid_pixel_count": 1_000_000,
    }


def run_one_epoch(
    runner: exact_runner.ExactRunner,
    components: dict[str, object],
) -> dict[str, object]:
    model = components["model"]
    optimizer = components["optimizer"]
    scaler = components["scaler"]
    generator = components["loader_generator"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is invalid")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer is invalid")
    if not isinstance(scaler, StateScaler):
        raise TypeError("fixture scaler is invalid")
    if not isinstance(generator, torch.Generator):
        raise TypeError("fixture generator is invalid")

    control = runner.next_epoch_control()
    for group in optimizer.param_groups:
        group["lr"] = control.learning_rate
    permutation = torch.randperm(7, generator=generator)
    python_value = random.getrandbits(31)
    numpy_value = int(np.random.randint(0, 2**31 - 1))
    inputs = torch.rand(2, 1, 4, 4)
    target = torch.tensor(
        [
            [(python_value % 101) / 101.0, (numpy_value % 103) / 103.0],
            [(numpy_value % 107) / 107.0, (python_value % 109) / 109.0],
        ]
    )
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    scaler.updates += 1
    loss_value = float(loss.detach())
    runner.commit_epoch(
        {
            "variant": V2_ON,
            "train_loss": loss_value,
            "processed_train_samples": 2,
            "epoch_seconds": 0.125,
            "loader_trace": permutation.tolist(),
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            **validation_metrics(control.epoch, loss_value),
        },
        extra_state={
            "variant": V2_ON,
            "relay_enabled": True,
            "relay_version": "v2_rms_centered_arctangent",
            "formal_eps": entry.FORMAL_EPS,
            "scaler_updates": scaler.updates,
        },
    )
    return {
        "epoch": control.epoch,
        "learning_rate": control.learning_rate,
        "permutation": permutation.tolist(),
        "python": python_value,
        "numpy": numpy_value,
        "loss": loss_value,
    }


def nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


class TPDNERV8MPRSDCHV2ExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_help_and_parser_expose_only_v2_on_and_frozen_axes(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                entry.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn(V2_ON, help_text)
        self.assertNotIn(entry.V1_RELAY_OFF_REFERENCE, help_text)

        args = parse(["--fresh"])
        self.assertEqual(args.variant, V2_ON)
        self.assertEqual(args.parent_variant, "tpd_clean_v8_mprs_dch_full")
        self.assertTrue(args.relay_enabled)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.split_seed, 20260722)
        self.assertEqual(args.epochs, 800)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.patch_size, 256)
        self.assertFalse(args.amp)
        self.assertEqual(args.run_tag, entry.FORMAL_RUN_TAG)
        self.assertEqual(
            entry.supported_candidate_variants(),
            (V2_ON,),
        )

        invalid = (
            ["--variant", entry.V1_RELAY_OFF_REFERENCE, "--fresh"],
            [
                "--variant",
                "tpd_ner_v8_mprs_dch_full_relay_on",
                "--fresh",
            ],
            ["--variant", "tpd_ner_v8_mprs_dch_v2_full_relay_off", "--fresh"],
            ["--variant", V2_ON, "--fresh", "--seed", "7"],
            ["--variant", V2_ON, "--fresh", "--batch-size", "8"],
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError, SystemExit)):
                    entry.parse_args(arguments)

    def test_production_builder_and_runtime_sources_are_v2_bound(self) -> None:
        model, metadata = entry.build_selected_model(V2_ON, 42)
        try:
            from model.tpd_ner_v8_mprs_dch_v2 import (
                TPDNERV8MPRSDCHV2SCTransNet,
            )

            self.assertIs(type(model), TPDNERV8MPRSDCHV2SCTransNet)
            manifest = metadata["architecture_manifest"]
            self.assertEqual(
                manifest["schema"],
                entry.ARCHITECTURE_MANIFEST_SCHEMA,
            )
            self.assertEqual(manifest["relay_state_key_count"], 16)
            self.assertEqual(manifest["relay_parameters"], 11_288)
            self.assertEqual(manifest["total_parameters"], 10_854_443)
            self.assertEqual(
                manifest["exact_resume_scope"],
                "same_v2_relay_on_variant_only",
            )
        finally:
            del model

        paths = entry.RUNTIME_SOURCE_PATHS
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertIn(
            entry.REPO_ROOT / "model/tpd_ner_v8_mprs_dch_v2.py",
            paths,
        )
        self.assertIn(
            entry.REPO_ROOT
            / "experiments/TPD_NER_V8_MPRS_DCH_V2_PROTOCOL.md",
            paths,
        )
        self.assertIn(
            entry.REPO_ROOT
            / "experiments/train_tpd_ner_v8_mprs_dch_exact.py",
            paths,
        )

    def test_temporary_source_lock_requires_complete_v2_sources(self) -> None:
        training_digest = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "variants": [V2_ON],
                "formal_contract": entry.formal_contract(),
                "training_data_sha256": training_digest,
                "source_sha256": {
                    str(runtime.relative_to(entry.REPO_ROOT)): (
                        entry.file_sha256(runtime)
                    )
                    for runtime in entry.RUNTIME_SOURCE_PATHS
                },
            }
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            contract = entry.source_lock_contract(training_digest, path)
            self.assertEqual(
                set(contract),
                {entry.SOURCE_LOCK_KEY, "training_data"},
            )

            missing = copy.deepcopy(payload)
            missing["source_sha256"].pop(
                "model/tpd_ner_v8_mprs_dch_v2.py"
            )
            path.write_text(json.dumps(missing, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "omits runtime sources"):
                entry.source_lock_contract(training_digest, path)

            wrong = copy.deepcopy(payload)
            wrong["schema"] = v1_exact.EXACT_SOURCE_LOCK_SCHEMA
            path.write_text(json.dumps(wrong, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                entry.source_lock_contract(training_digest, path)

    def test_identity_protocol_and_checkpoint_are_v2_only(self) -> None:
        with cpu_rng_environment():
            args = parse(["--fresh"])
            components = make_components(42)
            spec = make_spec(args, components)
            identity = exact_runner.build_run_identity(
                components["model"],
                spec,
            )
        validated = entry.require_v2_run_identity(
            identity,
            label="fixture",
            expected_variant=V2_ON,
        )
        self.assertTrue(validated["run_id"].startswith(entry.RUN_ID_PREFIX))
        determinism = validated["training_contract"]["determinism"]
        self.assertEqual(determinism["entry_schema"], entry.ENTRY_SCHEMA)
        self.assertEqual(
            determinism["relay_version"],
            "v2_rms_centered_arctangent",
        )
        self.assertFalse(determinism["gate_bias"])

        protocol = entry.protocol_payload(
            args,
            directory=Path("/tmp/v2-exact-fixture"),
            model_metadata=tiny_metadata(),
            normalization={"mean": 0.1, "std": 0.2},
            run_identity=identity,
        )
        self.assertEqual(protocol["schema"], entry.ENTRY_SCHEMA)
        self.assertEqual(
            protocol["comparison_design"]["required_control"],
            entry.V1_RELAY_OFF_REFERENCE,
        )
        self.assertFalse(
            protocol["comparison_design"]["relay_off_retrained"]
        )

        metrics = validation_metrics(1, 0.25)
        checkpoint = entry.EvaluatorCheckpointAdapter(
            model_metadata=tiny_metadata(),
            split_hashes={"train": "a" * 64},
        )(
            exact_runner.CompatibilityPayloadContext(
                role="last_evaluated_epoch",
                epoch=1,
                metrics=metrics,
                event={"epoch": 1, **metrics},
                exact_payload={
                    "model": {"state_dict": {"w": torch.tensor([1.0])}},
                    "optimizer": {"state_dict": {"state": {}}},
                    "scaler": {"state_dict": {"updates": 1}},
                },
                run_identity=identity,
                normalized_spec=spec.normalized(),
            )
        )
        self.assertEqual(checkpoint["schema"], entry.CHECKPOINT_SCHEMA)
        self.assertEqual(checkpoint["variant"], V2_ON)
        self.assertFalse(checkpoint["relay_off_retrained"])
        for name in (
            "derived_schema",
            "source_exact_checkpoint_sha256",
            "state_dict_sha256",
            "optimizer_state_sha256",
            "scaler_state_sha256",
        ):
            checkpoint[name] = "a" * 64
        entry.require_evaluator_checkpoint_payload(
            checkpoint,
            expected_variant=V2_ON,
        )

        v1_checkpoint = copy.deepcopy(checkpoint)
        v1_checkpoint["schema"] = v1_exact.CHECKPOINT_SCHEMA
        with self.assertRaisesRegex(ValueError, "schema differs"):
            entry.require_evaluator_checkpoint_payload(v1_checkpoint)
        wrong_variant = copy.deepcopy(identity)
        wrong_variant["variant"] = entry.V1_RELAY_OFF_REFERENCE
        with self.assertRaisesRegex(ValueError, "V1 relay variant"):
            entry.require_v2_run_identity(
                wrong_variant,
                label="wrong variant",
            )
        with self.assertRaises(ValueError):
            v1_exact._require_ner_run_identity(
                identity,
                label="v1 rejects v2",
            )

    def test_bad_journals_are_rejected_by_startup_before_any_state_restore(
        self,
    ) -> None:
        bad_identities = (
            {
                "variant": entry.V1_RELAY_OFF_REFERENCE,
                "run_id": "tpd-ner-v8-mprs-dch-exact:v1-off",
                "seed": 42,
                "split_seed": 20260722,
                "source_locks": {v1_exact.SOURCE_LOCK_KEY: "1" * 64},
                "training_contract": {
                    "determinism": {
                        "entry_schema": v1_exact.ENTRY_SCHEMA,
                    }
                },
            },
            {
                "variant": "tpd_ner_v8_mprs_dch_full_relay_on",
                "run_id": "tpd-ner-v8-mprs-dch-exact:v1-on",
                "seed": 42,
                "split_seed": 20260722,
                "source_locks": {v1_exact.SOURCE_LOCK_KEY: "1" * 64},
                "training_contract": {
                    "determinism": {
                        "entry_schema": v1_exact.ENTRY_SCHEMA,
                    }
                },
            },
            {
                "variant": "tpd_ner_v8_mprs_dch_v2_full_relay_off",
                "run_id": f"{entry.RUN_ID_PREFIX}wrong-v2-variant",
                "seed": 42,
                "split_seed": 20260722,
                "source_locks": {entry.SOURCE_LOCK_KEY: "1" * 64},
                "training_contract": {
                    "determinism": {
                        "entry_schema": entry.ENTRY_SCHEMA,
                    }
                },
            },
        )
        for bad_identity in bad_identities:
            with self.subTest(variant=bad_identity["variant"]):
                components = make_components(42)
                model = components["model"]
                optimizer = components["optimizer"]
                scaler = components["scaler"]
                generator = components["loader_generator"]
                model_before = copy.deepcopy(model.state_dict())
                optimizer_before = copy.deepcopy(optimizer.state_dict())
                scaler_before = copy.deepcopy(scaler.state_dict())
                rng_before = exact_resume.capture_rng_state(generator)

                runner = object.__new__(entry.TPDNERV8V2ExactRunner)
                runner.journal = mock.Mock()
                runner.journal.load_active.return_value = SimpleNamespace(
                    checkpoint_path=Path("/tmp/bad-exact.pth")
                )
                runner.spec = SimpleNamespace(variant=V2_ON)
                runner.model = model
                runner.optimizer = optimizer
                runner.scaler = scaler
                runner.loader_generator = generator
                runner._load_exact_payload = mock.Mock(
                    return_value=(
                        {
                            "run_identity": bad_identity,
                            "optimizer": {"state_dict": {}},
                        },
                        "digest",
                    )
                )
                with self.assertRaises(ValueError):
                    runner.startup(
                        exact_runner.InitializationRequest.exact()
                    )
                runner._load_exact_payload.assert_called_once()
                self.assertTrue(
                    nested_equal(model_before, model.state_dict())
                )
                self.assertTrue(
                    nested_equal(optimizer_before, optimizer.state_dict())
                )
                self.assertTrue(
                    nested_equal(scaler_before, scaler.state_dict())
                )
                self.assertTrue(
                    nested_equal(
                        rng_before,
                        exact_resume.capture_rng_state(generator),
                    )
                )

    def test_physical_gpu_identity_uses_v2_owned_gpu2_or_gpu3_binding(
        self,
    ) -> None:
        expected_uuid = entry.PHYSICAL_GPU_UUIDS["2"]
        shared_payload = {
            "device_uuid": expected_uuid,
            "cuda_visible_devices": expected_uuid,
        }
        with (
            mock.patch.object(
                entry.shared_exact,
                "environment_contract",
                return_value=copy.deepcopy(shared_payload),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX": "2",
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID": expected_uuid,
                },
                clear=False,
            ),
        ):
            payload = entry.environment_contract(torch.device("cuda:0"))
        self.assertEqual(payload["physical_gpu_index"], 2)
        self.assertEqual(payload["physical_gpu_uuid"], expected_uuid)
        self.assertEqual(
            payload["physical_gpu_assignment_source"],
            "verified_v2_ner_worker_environment",
        )

        with (
            mock.patch.object(
                entry.shared_exact,
                "environment_contract",
                return_value=copy.deepcopy(shared_payload),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX": "2",
                    "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID": "wrong",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "UUID differs"):
                entry.environment_contract(torch.device("cuda:0"))

    def test_continuous_four_equals_two_resume_two_all_state_exact(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse(["--fresh"])

            continuous_components = make_components(42)
            continuous_spec = make_spec(args, continuous_components)
            continuous = make_runner(
                root / "continuous",
                args,
                continuous_components,
                spec=continuous_spec,
            )
            continuous.startup(exact_runner.InitializationRequest.fresh())
            continuous_trace = [
                run_one_epoch(continuous, continuous_components)
                for _ in range(4)
            ]
            continuous_model = copy.deepcopy(
                continuous_components["model"].state_dict()
            )
            continuous_optimizer = copy.deepcopy(
                continuous_components["optimizer"].state_dict()
            )
            continuous_scaler = copy.deepcopy(
                continuous_components["scaler"].state_dict()
            )
            continuous_rng = exact_resume.capture_rng_state(
                continuous_components["loader_generator"]
            )
            continuous_metrics = (
                root / "continuous" / exact_runner.METRICS_FILENAME
            ).read_bytes()

            split_components = make_components(42)
            split_spec = make_spec(args, split_components)
            split = make_runner(
                root / "split",
                args,
                split_components,
                spec=split_spec,
            )
            split.startup(exact_runner.InitializationRequest.fresh())
            split_trace = [
                run_one_epoch(split, split_components)
                for _ in range(2)
            ]

            rebuilt_components = make_components(999)
            rebuilt_spec = make_spec(
                args,
                rebuilt_components,
                initial_model_state_sha256=(
                    split_spec.initial_model_state_sha256
                ),
                initial_rng=split_spec.initial_rng,
                policy=split_spec.selection_policy,
            )
            rebuilt = make_runner(
                root / "split",
                args,
                rebuilt_components,
                spec=rebuilt_spec,
            )
            restored = rebuilt.startup(
                exact_runner.InitializationRequest.exact()
            )
            self.assertEqual(restored.completed_epoch, 2)
            self.assertEqual(restored.next_epoch, 3)
            split_trace.extend(
                run_one_epoch(rebuilt, rebuilt_components)
                for _ in range(2)
            )

            self.assertEqual(continuous_trace, split_trace)
            self.assertEqual(
                continuous_metrics,
                (
                    root / "split" / exact_runner.METRICS_FILENAME
                ).read_bytes(),
            )
            self.assertTrue(
                nested_equal(
                    continuous_model,
                    rebuilt_components["model"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    continuous_optimizer,
                    rebuilt_components["optimizer"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    continuous_scaler,
                    rebuilt_components["scaler"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    continuous_rng,
                    exact_resume.capture_rng_state(
                        rebuilt_components["loader_generator"]
                    ),
                )
            )
            self.assertEqual(
                continuous.snapshot.best_selection,
                rebuilt.snapshot.best_selection,
            )
            self.assertEqual(rebuilt.snapshot.completed_epoch, 4)
            self.assertEqual(rebuilt.snapshot.next_epoch, 5)


if __name__ == "__main__":
    unittest.main()
