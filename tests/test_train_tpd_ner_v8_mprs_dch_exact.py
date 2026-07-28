from __future__ import annotations

import argparse
import copy
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact_resume
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_clean_v8_mprs_dch_exact as v8_entry
from experiments import train_tpd_ner_v8_mprs_dch as ordinary
from experiments import train_tpd_ner_v8_mprs_dch_exact as entry
from model.tpd_ner_v8_mprs_dch import (
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
)


OFF = "tpd_ner_v8_mprs_dch_full_relay_off"
ON = "tpd_ner_v8_mprs_dch_full_relay_on"


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


def parse(
    trailing: list[str],
    *,
    variant: str = ON,
) -> argparse.Namespace:
    return entry.parse_args(
        [
            "--variant",
            variant,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--batch-size",
            "2",
            "--patch-size",
            "32",
            "--base-lr",
            "0.004",
            "--min-lr",
            "0.0001",
            "--warmup-epochs",
            "1",
            *trailing,
        ]
    )


def make_components(seed: int) -> dict[str, object]:
    seed_all(seed)
    model = TinyTrajectoryModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.004,
        betas=(0.8, 0.95),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(entry.TRAINING_SEED)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": generator,
    }


def tiny_metadata(variant: str) -> dict[str, object]:
    candidate = entry.candidate_contract(variant)
    manifest = {
        "schema": entry.ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": "tests.TinyTrajectoryModel",
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": entry.RELAY_WIDTH,
        "eps": entry.FORMAL_EPS,
    }
    return {
        "variant": variant,
        "candidate_family": "fixture-v8-mprs-dch-ner",
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": entry.RELAY_WIDTH,
        "relay_initialization_seed": entry.RELAY_INITIALIZATION_SEED,
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
    scaler = components["scaler"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is not an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer has the wrong type")
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=tiny_metadata(args.variant),
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=(
            exact_runner.fresh_initialization_contract()
        ),
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
                "train",
                ("a", "b"),
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation",
                ("c",),
            ),
        },
        data_records={
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples",
                ("a:image", "b:image"),
            ),
            "normalization": exact_runner.OrderedFingerprint.from_values(
                "normalization",
                ('{"mean":1.0,"std":2.0}',),
            ),
        },
        environment={"name": "cpu-ner-v8-fixture"},
    )


def make_runner(
    directory: Path,
    args: argparse.Namespace,
    components: dict[str, object],
    *,
    spec: exact_runner.ExactRunSpec | None = None,
) -> entry.TPDNERV8ExactRunner:
    model = components["model"]
    optimizer = components["optimizer"]
    generator = components["loader_generator"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is not an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer has the wrong type")
    if not isinstance(generator, torch.Generator):
        raise TypeError("fixture loader generator has the wrong type")
    return entry.TPDNERV8ExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=components["scaler"],
        loader_generator=generator,
        spec=spec or make_spec(args, components),
        selection_policy=selection_policy(),
        compatibility_payload_factory=entry.EvaluatorCheckpointAdapter(
            model_metadata=tiny_metadata(args.variant),
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
    variant: str,
) -> dict[str, object]:
    model = components["model"]
    optimizer = components["optimizer"]
    scaler = components["scaler"]
    generator = components["loader_generator"]
    if not isinstance(model, nn.Module):
        raise TypeError("fixture model is not an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("fixture optimizer has the wrong type")
    if not isinstance(scaler, StateScaler):
        raise TypeError("fixture scaler has the wrong type")
    if not isinstance(generator, torch.Generator):
        raise TypeError("fixture generator has the wrong type")

    control = runner.next_epoch_control()
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
            "variant": variant,
            "train_loss": loss_value,
            "processed_train_samples": 2,
            "epoch_seconds": 0.125,
            "loader_trace": permutation.tolist(),
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            **validation_metrics(control.epoch, loss_value),
        },
        extra_state={
            "variant": variant,
            "relay_enabled": entry.candidate_contract(variant)[
                "relay_enabled"
            ],
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


class TPDNERV8MPRSDCHExactTests(unittest.TestCase):
    def test_help_is_standalone_and_lists_only_ner_combinations(self) -> None:
        for arguments in (
            ["--help"],
            ["--variant", ON, "--help"],
        ):
            with self.subTest(arguments=arguments):
                output = io.StringIO()
                with redirect_stdout(output):
                    with self.assertRaises(SystemExit) as raised:
                        entry.parse_args(arguments)
                self.assertEqual(raised.exception.code, 0)
                help_text = output.getvalue()
                self.assertIn(
                    "Exact-resume V8-MPRS-DCH five-node NER training",
                    help_text,
                )
                self.assertIn(OFF, help_text)
                self.assertIn(ON, help_text)
                self.assertNotIn("tpd_clean_v7_dch_", help_text)

    def test_parse_freezes_axes_and_combination_identity(self) -> None:
        self.assertEqual(
            entry.supported_candidate_variants(),
            ordinary.SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS,
        )
        for variant, enabled in ((OFF, False), (ON, True)):
            with self.subTest(variant=variant):
                args = parse(["--fresh"], variant=variant)
                self.assertEqual(args.variant, variant)
                self.assertEqual(
                    args.parent_variant,
                    "tpd_clean_v8_mprs_dch_full",
                )
                self.assertIs(args.relay_enabled, enabled)
                self.assertEqual(args.seed, 42)
                self.assertEqual(args.split_seed, 20260722)
                self.assertEqual(args.relay_width, 8)
                self.assertEqual(args.relay_initialization_seed, 42)
                self.assertEqual(
                    entry.run_directory(args).parts[-2],
                    variant,
                )

        invalid_axes = (
            ["--fresh", "--seed", "3407"],
            ["--fresh", "--split-seed", "9"],
            ["--fresh", "--relay-enabled"],
            ["--fresh", "--relay-off"],
            ["--fresh", "--relay-width", "16"],
            ["--fresh", "--relay-initialization-seed", "7"],
        )
        for trailing in invalid_axes:
            with self.subTest(trailing=trailing):
                with self.assertRaises((ValueError, SystemExit)):
                    parse(trailing)

    def test_production_builder_matches_both_combination_contracts(self) -> None:
        for variant, enabled, relay_parameters, total_parameters in (
            (OFF, False, 0, PRODUCTION_PARENT_PARAMETERS),
            (
                ON,
                True,
                PRODUCTION_RELAY_PARAMETERS,
                PRODUCTION_RELAY_ON_PARAMETERS,
            ),
        ):
            with self.subTest(variant=variant):
                model, metadata = entry.build_selected_model(variant, 42)
                self.assertIs(model.relay_enabled, enabled)
                self.assertEqual(metadata["variant"], variant)
                self.assertIs(metadata["relay_enabled"], enabled)
                manifest = metadata["architecture_manifest"]
                self.assertEqual(
                    manifest["schema"],
                    entry.ARCHITECTURE_MANIFEST_SCHEMA,
                )
                self.assertEqual(manifest["relay_parameters"], relay_parameters)
                self.assertEqual(manifest["total_parameters"], total_parameters)
                self.assertEqual(
                    manifest["exact_resume_scope"],
                    "same_combination_variant_only",
                )
                self.assertEqual(
                    metadata["architecture_id"],
                    entry.canonical_sha256(manifest),
                )
                self.assertNotEqual(
                    metadata["architecture_id"],
                    metadata["ordinary_builder_architecture_id"],
                )

    def test_import_is_side_effect_free_and_reuses_verified_loop(self) -> None:
        self.assertIs(
            entry._reused_training_kernel().__code__,
            v8_entry._reused_training_kernel().__code__,
        )
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(entry.REPO_ROOT)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from experiments import "
                        "train_tpd_ner_v8_mprs_dch_exact as e;"
                        "print(e.ENTRY_SCHEMA)"
                    ),
                ],
                cwd=directory,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stdout.strip(), entry.ENTRY_SCHEMA)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_identity_protocol_checkpoint_and_scheduler_are_combination_bound(
        self,
    ) -> None:
        with cpu_rng_environment():
            args = parse(["--fresh"], variant=ON)
            components = make_components(42)
            spec = make_spec(args, components)
            identity = exact_runner.build_run_identity(
                components["model"],
                spec,
            )
        normalized = spec.normalized()
        self.assertEqual(normalized["variant"], ON)
        self.assertNotIn(":relay-on:", normalized["run_id"])
        self.assertEqual(
            normalized["determinism"]["parent_variant"],
            "tpd_clean_v8_mprs_dch_full",
        )
        self.assertIs(normalized["determinism"]["relay_enabled"], True)
        self.assertEqual(
            normalized["determinism"]["scheduler_restore_mode"],
            "identity_bound_manual_schedule_from_completed_epoch",
        )

        protocol = entry.protocol_payload(
            args,
            directory=Path("/tmp/ner-v8-exact-fixture"),
            model_metadata=tiny_metadata(ON),
            normalization={"mean": 0.1, "std": 0.2},
            run_identity=identity,
        )
        self.assertEqual(protocol["schema"], entry.ENTRY_SCHEMA)
        self.assertEqual(
            protocol["exact_resume_policy"]["same_version"],
            "same_combination_variant_epoch_boundary_only",
        )
        self.assertIs(protocol["relay_identity"]["enabled"], True)

        metrics = validation_metrics(1, 0.25)
        checkpoint = entry.EvaluatorCheckpointAdapter(
            model_metadata=tiny_metadata(ON),
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
                normalized_spec=normalized,
            )
        )
        required_checkpoint_fields = {
            "schema",
            "variant",
            "dataset",
            "seed",
            "split_seed",
            "checkpoint_role",
            "state_dict",
            "model_metadata",
            "run_identity",
            "checkpoint_identity",
            "validation_metrics",
        }
        self.assertTrue(
            required_checkpoint_fields.issubset(checkpoint)
        )
        self.assertEqual(
            set(entry.EVALUATOR_CHECKPOINT_REQUIRED_FIELDS),
            set(checkpoint),
        )
        self.assertEqual(checkpoint["schema"], entry.CHECKPOINT_SCHEMA)
        self.assertEqual(checkpoint["variant"], ON)
        self.assertIs(checkpoint["relay_enabled"], True)
        self.assertEqual(
            checkpoint["scheduler"],
            {
                "kind": "identity_bound_manual_schedule",
                "completed_epoch": 1,
            },
        )
        self.assertEqual(
            checkpoint["checkpoint_identity"]["schema"],
            entry.CHECKPOINT_IDENTITY_SCHEMA,
        )
        self.assertTrue(
            checkpoint["run_identity"]["run_id"].startswith(
                entry.RUN_ID_PREFIX
            )
        )
        invalid_checkpoint = copy.deepcopy(checkpoint)
        invalid_checkpoint.pop("state_dict")
        with self.assertRaisesRegex(ValueError, "lacks fields"):
            entry.require_evaluator_checkpoint_payload(
                invalid_checkpoint,
                expected_variant=ON,
            )
        invalid_checkpoint = copy.deepcopy(checkpoint)
        invalid_checkpoint["scheduler"]["completed_epoch"] = 2
        with self.assertRaisesRegex(ValueError, "scheduler differs"):
            entry.require_evaluator_checkpoint_payload(
                invalid_checkpoint,
                expected_variant=ON,
            )
        invalid_checkpoint = copy.deepcopy(checkpoint)
        invalid_checkpoint["selection_source"] = "other"
        with self.assertRaisesRegex(ValueError, "selection source"):
            entry.require_evaluator_checkpoint_payload(
                invalid_checkpoint,
                expected_variant=ON,
            )

    def test_runtime_sources_and_temporary_source_lock_are_complete(self) -> None:
        runtime_paths = tuple(entry.RUNTIME_SOURCE_PATHS)
        self.assertEqual(len(runtime_paths), len(set(runtime_paths)))
        self.assertTrue(all(path.is_file() for path in runtime_paths))
        self.assertIn(
            entry.REPO_ROOT / "model/tpd_clean.py",
            runtime_paths,
        )
        training_digest = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "temporary-ner-exact-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "variants": list(entry.supported_candidate_variants()),
                "formal_contract": entry.formal_contract(),
                "training_data_sha256": training_digest,
                "source_sha256": {
                    str(path.relative_to(entry.REPO_ROOT)): (
                        entry.file_sha256(path)
                    )
                    for path in runtime_paths
                },
            }
            lock_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            contract = entry.source_lock_contract(
                training_digest,
                lock_path,
            )
            self.assertEqual(contract["training_data"], training_digest)
            self.assertEqual(
                set(contract),
                {entry.SOURCE_LOCK_KEY, "training_data"},
            )

            relative = "model/tpd_clean.py"
            payload["source_sha256"][relative] = "0" * 64
            lock_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, relative):
                entry.source_lock_contract(training_digest, lock_path)

    def test_continuous_and_epoch_boundary_resume_are_tensor_exact(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse(["--fresh"], variant=ON)

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
                run_one_epoch(continuous, continuous_components, ON)
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
                run_one_epoch(split, split_components, ON)
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
                run_one_epoch(rebuilt, rebuilt_components, ON)
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

    def test_completion_summary_carries_exact_run_identity(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse(["--fresh"], variant=OFF)
            components = make_components(42)
            spec = make_spec(args, components)
            identity = exact_runner.build_run_identity(
                components["model"],
                spec,
            )
            events = []
            for epoch in range(1, entry.FORMAL_EPOCHS + 1):
                metrics = validation_metrics(min(epoch, 4), 0.25)
                events.append(
                    {
                        "epoch": epoch,
                        "epoch_seconds": 0.1,
                        "skipped_singleton_batches": 0,
                        **metrics,
                    }
                )
            (root / exact_runner.METRICS_FILENAME).write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": entry.ENTRY_SCHEMA,
                        "run_identity": identity,
                    }
                ),
                encoding="utf-8",
            )
            summary = entry.completion_summary(
                args,
                directory=root,
                model_metadata=tiny_metadata(OFF),
                split_hashes={"train": "a" * 64},
                selection={
                    "primary": {"epoch": 1},
                    "secondary": {"epoch": 1},
                },
            )
        self.assertEqual(
            summary["schema"],
            entry.COMPLETION_SUMMARY_SCHEMA,
        )
        self.assertEqual(summary["run_identity"], identity)
        self.assertEqual(summary["variant"], OFF)
        self.assertEqual(summary["split_seed"], 20260722)
        self.assertIs(summary["relay_enabled"], False)

    def test_cross_combination_journal_rejected_before_state_restore(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            off_args = parse(["--fresh"], variant=OFF)
            off_components = make_components(42)
            off_spec = make_spec(off_args, off_components)
            off_runner = make_runner(
                root,
                off_args,
                off_components,
                spec=off_spec,
            )
            off_runner.startup(exact_runner.InitializationRequest.fresh())
            run_one_epoch(off_runner, off_components, OFF)

            on_args = parse(["--fresh"], variant=ON)
            on_components = make_components(999)
            on_spec = make_spec(on_args, on_components)
            on_runner = make_runner(
                root,
                on_args,
                on_components,
                spec=on_spec,
            )
            model_before = copy.deepcopy(
                on_components["model"].state_dict()
            )
            optimizer_before = copy.deepcopy(
                on_components["optimizer"].state_dict()
            )
            scaler_before = copy.deepcopy(
                on_components["scaler"].state_dict()
            )
            rng_before = exact_resume.capture_rng_state(
                on_components["loader_generator"]
            )
            with self.assertRaisesRegex(ValueError, "differs from requested"):
                on_runner.startup(
                    exact_runner.InitializationRequest.exact()
                )
            self.assertTrue(
                nested_equal(
                    model_before,
                    on_components["model"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    optimizer_before,
                    on_components["optimizer"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    scaler_before,
                    on_components["scaler"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    rng_before,
                    exact_resume.capture_rng_state(
                        on_components["loader_generator"]
                    ),
                )
            )

    def test_v7_and_v8_only_protocols_are_rejected(self) -> None:
        args = parse(["--exact-resume"], variant=ON)
        cases = (
            (
                v8_entry.ENTRY_SCHEMA,
                "V8 tokenizer-only protocol",
            ),
            (
                v8_entry.exact_kernel.ENTRY_SCHEMA,
                "V7 protocol",
            ),
            (
                entry.ENTRY_SCHEMA,
                "entry schema",
            ),
        )
        for schema, message in cases:
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "protocol.json").write_text(
                    json.dumps({"schema": schema, "run_identity": {}}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    entry.initialization_plan(
                        args,
                        root,
                        TinyTrajectoryModel(),
                    )


if __name__ == "__main__":
    unittest.main()
