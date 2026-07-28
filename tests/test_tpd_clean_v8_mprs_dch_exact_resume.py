from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact_resume
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_clean_v7_dch_exact as v7_entry
from experiments import train_tpd_clean_v8_mprs_dch_exact as entry
from experiments import train_tpd_pilot as base
from model.tpd_clean_v8_mprs_dch import (
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
)


VARIANT = "tpd_clean_v8_mprs_dch_full"
V8_BLOCK = (
    "model.tpd_clean_v8_mprs_dch.TPDCleanV8MPRSDCHBlock"
)


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
        self.block = TPDCleanV8MPRSDCHBlock(
            1,
            activate=False,
            context_gate=1.0,
        )
        self.head = nn.Linear(4, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(value).flatten(1))


class TinyEmbedding(nn.Module):
    def __init__(self, count: int, *, context_gate: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TPDCleanV8MPRSDCHBlock(
                1,
                activate=index < count - 1,
                context_gate=context_gate,
            )
            for index in range(count)
        )


class TinyArchitecture(nn.Module):
    def __init__(self, *, context_gate: float) -> None:
        super().__init__()
        self.mtc = nn.Module()
        self.mtc.embeddings_1 = TinyEmbedding(
            4,
            context_gate=context_gate,
        )
        self.mtc.embeddings_2 = TinyEmbedding(
            3,
            context_gate=context_gate,
        )


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
    variant: str = VARIANT,
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


def tiny_args() -> argparse.Namespace:
    return parse(["--fresh"])


def make_components(seed: int) -> dict[str, object]:
    seed_all(seed)
    model = TinyTrajectoryModel()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.004,
        betas=(0.8, 0.95),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": generator,
    }


def tiny_metadata() -> dict[str, object]:
    return {
        "variant": VARIANT,
        "candidate_family": "fixture-v8-mprs-dch",
        "architecture_manifest": {
            "schema": entry.ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": VARIANT,
            "model": "tests.TinyTrajectoryModel",
            "block": V8_BLOCK,
            "eps": entry.FORMAL_EPS,
        },
    }


def builder_metadata(variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "variant_spec": {
            "state_compatible_with": "tpd_clean_v7_dch",
        },
        "candidate_family": "fixture-v8-mprs-dch",
        "mainline_contract": "Keep-Context-Saliency",
        "semantic_sources": ("Keep", "Context", "Saliency"),
        "replaced_embeddings": (
            "mtc.embeddings_1",
            "mtc.embeddings_2",
        ),
        "phase_order": (
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ),
        "pixel_unshuffle_channel_order": (
            "input_channel_major_four_phases_contiguous"
        ),
        "saliency_representation": "mass_preserving_phase_resolved",
        "saliency_source_equation": "S_p=S0+(Z_p-C0)/3",
        "saliency_mass_equation": "sum_p(S_p)=4*S0",
        "saliency_nonnegative": True,
        "saliency_projection": "complete_keep_weight_phase_projection",
        "saliency_reuse_equation": "Sa8=Sa7+((K-b)-Ca)/3",
        "phase_tied_projection_formula": "Wt=sum_phase(Wk)",
        "context_code_formula": "Q=tanh(centered/rms_eps)",
        "context_headroom_formula": "H=1+abs(a)*(1-abs(a))*V",
        "fusion_equation": "K+Sa8*(a*H)",
        "zero_scale_first_order_reference": "capacity_exact",
        "phase_contrast_parameters": 0,
        "phase_contrast_buffers": 0,
        "derived_projection_parameters": 0,
        "derived_projection_buffers": 0,
        "shallow_embedding_parameters": 66_176,
        "total_parameters": 10_843_155,
        "state_compatible_with": "tpd_clean_v7_dch",
        "cross_version_exact_resume_supported": False,
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
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=tiny_metadata(),
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
            "validation": (
                exact_runner.OrderedFingerprint.from_values(
                    "validation",
                    ("c",),
                )
            ),
        },
        data_records={
            "train_samples": (
                exact_runner.OrderedFingerprint.from_values(
                    "train_samples",
                    ("a:image", "b:image"),
                )
            ),
            "normalization": (
                exact_runner.OrderedFingerprint.from_values(
                    "normalization",
                    ('{"mean":1.0,"std":2.0}',),
                )
            ),
        },
        environment={"name": "cpu-v8-fixture"},
    )


def make_runner(
    directory: Path,
    args: argparse.Namespace,
    components: dict[str, object],
    *,
    spec: exact_runner.ExactRunSpec | None = None,
) -> entry.MPRSDCHExactRunner:
    model = components["model"]
    optimizer = components["optimizer"]
    generator = components["loader_generator"]
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(generator, torch.Generator)
    return entry.MPRSDCHExactRunner(
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
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(scaler, StateScaler)
    assert isinstance(generator, torch.Generator)

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
            "variant": VARIANT,
            "train_loss": loss_value,
            "processed_train_samples": 2,
            "epoch_seconds": 0.125,
            "loader_trace": permutation.tolist(),
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            **validation_metrics(control.epoch, loss_value),
        },
        extra_state={
            "variant": VARIANT,
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


class TPDCleanV8MPRSDCHExactResumeTests(unittest.TestCase):
    def test_import_has_no_side_effect_and_reuses_loop_code(self) -> None:
        self.assertIsNot(
            base.build_model,
            entry.ordinary.build_clean_v8_mprs_dch_model,
        )
        self.assertNotEqual(
            tuple(base.SUPPORTED_VARIANTS),
            SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
        )
        self.assertIs(
            entry._reused_training_kernel().__code__,
            v7_entry.run_training.__code__,
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
                        "train_tpd_clean_v8_mprs_dch_exact as entry;"
                        "from experiments import train_tpd_pilot as base;"
                        "assert base.build_model is not "
                        "entry.ordinary.build_clean_v8_mprs_dch_model;"
                        "print(entry.ENTRY_SCHEMA)"
                    ),
                ],
                cwd=directory,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.stdout.strip(),
                entry.ENTRY_SCHEMA,
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_builder_and_all_owned_identities_are_v8_bound(self) -> None:
        for variant in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
            gate = 1.0 if variant.endswith("_full") else 0.0
            model = TinyArchitecture(context_gate=gate)
            with mock.patch.object(
                entry.ordinary,
                "build_clean_v8_mprs_dch_model",
                return_value=(model, builder_metadata(variant)),
            ):
                selected, metadata = entry.build_selected_model(
                    variant,
                    42,
                )
            self.assertIs(selected, model)
            manifest = metadata["architecture_manifest"]
            self.assertEqual(
                manifest["schema"],
                entry.ARCHITECTURE_MANIFEST_SCHEMA,
            )
            self.assertEqual(manifest["variant"], variant)
            self.assertEqual(manifest["block"], V8_BLOCK)
            self.assertFalse(
                manifest["cross_version_exact_resume_supported"]
            )
            self.assertNotIn(
                "v7",
                json.dumps(metadata, sort_keys=True).lower(),
            )

        with cpu_rng_environment():
            args = tiny_args()
            components = make_components(42)
            spec = make_spec(args, components)
            identity = exact_runner.build_run_identity(
                components["model"],
                spec,
            )
        normalized = spec.normalized()
        self.assertTrue(normalized["run_id"].startswith(entry.RUN_ID_PREFIX))
        self.assertEqual(normalized["variant"], VARIANT)
        self.assertEqual(
            normalized["determinism"]["entry_schema"],
            entry.ENTRY_SCHEMA,
        )
        self.assertEqual(
            set(normalized["source_locks"]),
            {entry.SOURCE_LOCK_KEY},
        )

        protocol = entry.protocol_payload(
            args,
            directory=Path("/tmp/v8-exact-fixture"),
            model_metadata=tiny_metadata(),
            normalization={"mean": 0.1, "std": 0.2},
            run_identity=identity,
        )
        self.assertEqual(protocol["schema"], entry.ENTRY_SCHEMA)
        self.assertEqual(
            protocol["exact_resume_policy"][
                "cross_version_optimizer_journal"
            ],
            "forbidden",
        )

        metrics = validation_metrics(1, 0.25)
        adapter = entry.EvaluatorCheckpointAdapter(
            model_metadata=tiny_metadata(),
            split_hashes={"train": "a" * 64},
        )
        checkpoint = adapter(
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
        self.assertEqual(checkpoint["variant"], VARIANT)
        self.assertEqual(
            checkpoint["checkpoint_identity"]["schema"],
            entry.CHECKPOINT_IDENTITY_SCHEMA,
        )
        self.assertEqual(
            checkpoint["checkpoint_identity"]["run_id"],
            identity["run_id"],
        )

        owned_strings = (
            entry.ENTRY_SCHEMA,
            entry.EXACT_SOURCE_LOCK_SCHEMA,
            entry.ARCHITECTURE_MANIFEST_SCHEMA,
            entry.CHECKPOINT_IDENTITY_SCHEMA,
            entry.COMPLETION_SUMMARY_SCHEMA,
            entry.SOURCE_LOCK_KEY,
            normalized["run_id"],
            normalized["variant"],
            normalized["determinism"]["entry_schema"],
            checkpoint["checkpoint_identity"]["schema"],
            checkpoint["checkpoint_identity"]["run_id"],
            V8_BLOCK,
        )
        for value in owned_strings:
            with self.subTest(value=value):
                self.assertNotIn("v7", value.lower())

    def test_temporary_source_lock_has_only_v8_owned_key(self) -> None:
        training_digest = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "temporary-v8-exact-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "variants": list(
                    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS
                ),
                "formal_contract": entry.formal_contract(),
                "training_data_sha256": training_digest,
                "source_sha256": {
                    str(path.relative_to(entry.REPO_ROOT)): (
                        entry.file_sha256(path)
                    )
                    for path in entry.RUNTIME_SOURCE_PATHS
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
            self.assertIn(entry.SOURCE_LOCK_KEY, contract)
            self.assertNotIn(
                "tpd_clean_v7_dch_exact_source_lock",
                contract,
            )
            self.assertNotIn(
                "v7",
                json.dumps(contract, sort_keys=True).lower(),
            )

            payload["schema"] = v7_entry.EXACT_SOURCE_LOCK_SCHEMA
            lock_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                entry.source_lock_contract(training_digest, lock_path)

    def test_environment_contract_uses_v8_owned_gpu_assignment(self) -> None:
        gpu_index = "2"
        gpu_uuid = entry.PHYSICAL_GPU_UUIDS[gpu_index]
        shared_payload = {
            "device_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
        }
        with (
            mock.patch.object(
                entry.shared_exact,
                "environment_contract",
                return_value=dict(shared_payload),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX": gpu_index,
                    "TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID": gpu_uuid,
                },
                clear=False,
            ),
        ):
            observed = entry.environment_contract(
                torch.device("cuda:0")
            )
        self.assertEqual(observed["physical_gpu_index"], 2)
        self.assertEqual(observed["physical_gpu_uuid"], gpu_uuid)
        self.assertEqual(
            observed["physical_gpu_assignment_source"],
            "verified_v8_worker_environment",
        )

        with (
            mock.patch.object(
                entry.shared_exact,
                "environment_contract",
                return_value=dict(shared_payload),
            ),
            mock.patch.dict(
                os.environ,
                {
                    "TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX": gpu_index,
                    "TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID": "wrong",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "PHYSICAL_GPU_UUID",
            ):
                entry.environment_contract(torch.device("cuda:0"))

    def test_v8_to_v8_epoch_boundary_resume_is_tensor_exact(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()

            continuous_components = make_components(42)
            continuous_spec = make_spec(args, continuous_components)
            continuous = make_runner(
                root / "continuous",
                args,
                continuous_components,
                spec=continuous_spec,
            )
            continuous.startup(
                exact_runner.InitializationRequest.fresh()
            )
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
                root
                / "continuous"
                / exact_runner.METRICS_FILENAME
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

            payload = torch.load(
                rebuilt.snapshot.active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["epoch"], 4)
            self.assertEqual(payload["run_identity"]["variant"], VARIANT)
            self.assertTrue(
                payload["run_identity"]["run_id"].startswith(
                    entry.RUN_ID_PREFIX
                )
            )
            last = torch.load(
                root / "split" / exact_runner.LAST_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                last["checkpoint_identity"]["schema"],
                entry.CHECKPOINT_IDENTITY_SCHEMA,
            )
            self.assertEqual(last["variant"], VARIANT)

    def test_v7_protocol_is_rejected_before_resume_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": v7_entry.ENTRY_SCHEMA,
                        "run_identity": {},
                    }
                ),
                encoding="utf-8",
            )
            args = parse(["--exact-resume"])
            with self.assertRaisesRegex(
                ValueError,
                "V7 protocol",
            ):
                entry.initialization_plan(
                    args,
                    root,
                    TinyTrajectoryModel(),
                )

    def test_v7_journal_is_rejected_before_optimizer_or_rng_load(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()
            v7_components = make_components(42)
            v8_spec = make_spec(args, v7_components)
            v7_determinism = dict(v8_spec.determinism)
            v7_determinism["entry_schema"] = v7_entry.ENTRY_SCHEMA
            v7_spec = replace(
                v8_spec,
                run_id=(
                    "tpd-clean-v7-dch-exact:fixture:"
                    "tpd_clean_v7_dch_full:seed-42:test"
                ),
                variant="tpd_clean_v7_dch_full",
                source_locks={
                    "tpd_clean_v7_dch_exact_source_lock": "2" * 64
                },
                determinism=v7_determinism,
            )
            v7_runner = v7_entry.DCHExactRunner(
                root,
                model=v7_components["model"],
                optimizer=v7_components["optimizer"],
                scaler=v7_components["scaler"],
                loader_generator=v7_components["loader_generator"],
                spec=v7_spec,
                selection_policy=selection_policy(),
            )
            v7_runner.startup(
                exact_runner.InitializationRequest.fresh()
            )
            run_one_epoch(v7_runner, v7_components)

            rebuilt_components = make_components(999)
            rebuilt_spec = make_spec(
                args,
                rebuilt_components,
                initial_model_state_sha256=(
                    v8_spec.initial_model_state_sha256
                ),
                initial_rng=v8_spec.initial_rng,
                policy=v8_spec.selection_policy,
            )
            rebuilt = make_runner(
                root,
                args,
                rebuilt_components,
                spec=rebuilt_spec,
            )
            model_before = copy.deepcopy(
                rebuilt_components["model"].state_dict()
            )
            optimizer_before = copy.deepcopy(
                rebuilt_components["optimizer"].state_dict()
            )
            scaler_before = copy.deepcopy(
                rebuilt_components["scaler"].state_dict()
            )
            rng_before = exact_resume.capture_rng_state(
                rebuilt_components["loader_generator"]
            )
            with self.assertRaisesRegex(
                ValueError,
                "V7 optimizer/journal",
            ):
                rebuilt.startup(
                    exact_runner.InitializationRequest.exact()
                )
            self.assertTrue(
                nested_equal(
                    model_before,
                    rebuilt_components["model"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    optimizer_before,
                    rebuilt_components["optimizer"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    scaler_before,
                    rebuilt_components["scaler"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    rng_before,
                    exact_resume.capture_rng_state(
                        rebuilt_components["loader_generator"]
                    ),
                )
            )

    def test_completion_summary_schema_is_v8_owned(self) -> None:
        args = tiny_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {
                    "epoch": epoch,
                    "epoch_seconds": 0.1,
                    "skipped_singleton_batches": 0,
                    **validation_metrics(epoch, 0.3 / epoch),
                }
                for epoch in (1, 2)
            ]
            (root / exact_runner.METRICS_FILENAME).write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with mock.patch.object(entry, "FORMAL_EPOCHS", 2):
                summary = entry.completion_summary(
                    args,
                    directory=root,
                    model_metadata=tiny_metadata(),
                    split_hashes={"train": "a" * 64},
                    selection={
                        "primary": {"epoch": 1},
                        "secondary": {"epoch": 2},
                    },
                )
        self.assertEqual(
            summary["schema"],
            entry.COMPLETION_SUMMARY_SCHEMA,
        )
        self.assertEqual(summary["variant"], VARIANT)
        self.assertNotIn("v7", summary["schema"].lower())
        self.assertEqual(
            tuple(summary["best_pd_validation_metrics"]),
            entry.STORED_VALIDATION_METRICS,
        )
        self.assertEqual(
            tuple(summary["best_miou_validation_metrics"]),
            entry.STORED_VALIDATION_METRICS,
        )


if __name__ == "__main__":
    unittest.main()
