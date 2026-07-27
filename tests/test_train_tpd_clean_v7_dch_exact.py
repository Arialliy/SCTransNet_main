from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact_resume
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_clean_v7_dch_exact as entry
from experiments import train_tpd_pilot as base
from model.tpd_clean_v7_dch import (
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
    TPDCleanV7DCHBlock,
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


def parse(arguments: list[str]) -> argparse.Namespace:
    return entry.parse_args(
        [
            "--variant",
            "tpd_clean_v7_dch_full",
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--batch-size",
            "2",
            "--patch-size",
            "32",
            *arguments,
        ]
    )


def tiny_args() -> argparse.Namespace:
    return parse(
        [
            "--base-lr",
            "0.004",
            "--min-lr",
            "0.0001",
            "--warmup-epochs",
            "1",
            "--fresh",
        ]
    )


def tiny_components(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Tanh(),
        nn.Dropout(p=0.1),
        nn.Linear(5, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return model, optimizer, scaler, generator


def tiny_metadata() -> dict:
    return {
        "variant": "tpd_clean_v7_dch_full",
        "candidate_family": "fixture-dch",
        "architecture_manifest": {
            "schema": entry.ARCHITECTURE_MANIFEST_SCHEMA,
            "model": "tiny",
            "block": "fixture",
            "eps": entry.FORMAL_EPS,
        },
    }


def selection_policy() -> exact_runner.SelectionPolicy:
    return exact_runner.pd_miou_selection_policy(
        stored_metrics=entry.STORED_VALIDATION_METRICS
    )


def tiny_spec(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    initial_model_state_sha256: str | None = None,
    initial_rng: dict | None = None,
    policy: dict | None = None,
) -> exact_runner.ExactRunSpec:
    split_records = {
        "train": exact_runner.OrderedFingerprint.from_values(
            "train",
            ("a", "b"),
        ),
        "validation": exact_runner.OrderedFingerprint.from_values(
            "validation",
            ("c",),
        ),
    }
    data_records = {
        "train_samples": exact_runner.OrderedFingerprint.from_values(
            "train_samples",
            ("a:image", "b:image"),
        ),
        "normalization": exact_runner.OrderedFingerprint.from_values(
            "normalization",
            ('{"mean":1.0,"std":2.0}',),
        ),
    }
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=tiny_metadata(),
        optimizer=optimizer,
        scaler=scaler,
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
        source_locks={"fixture": "1" * 64},
        split_records=split_records,
        data_records=data_records,
        environment={"name": "cpu-fixture"},
    )


def tiny_builder_metadata(variant: str) -> dict:
    return {
        "variant": variant,
        "candidate_family": "fixture-dch",
        "mainline_contract": "Keep-Context-Saliency",
        "semantic_sources": ("Keep", "Context", "Saliency"),
        "replaced_embeddings": ("mtc.embeddings_1", "mtc.embeddings_2"),
        "phase_tied_projection_formula": "Wt=sum_phase(Wk)",
        "context_code_formula": "Q=tanh(centered/rms_eps)",
        "context_headroom_formula": "H=1+abs(a)*(1-abs(a))*V",
        "fusion_equation": "K+Sa*(a*H)",
        "zero_scale_first_order_reference": "capacity_exact",
        "derived_projection_parameters": 0,
        "derived_projection_buffers": 0,
        "shallow_embedding_parameters": 66_176,
        "total_parameters": 10_843_155,
    }


class TinyEmbedding(nn.Module):
    def __init__(self, count: int, *, context_gate: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TPDCleanV7DCHBlock(
                1,
                activate=index < count - 1,
                context_gate=context_gate,
            )
            for index in range(count)
        )


class TinyDCH(nn.Module):
    def __init__(self, *, context_gate: float) -> None:
        super().__init__()
        self.mtc = nn.Module()
        self.mtc.embeddings_1 = TinyEmbedding(4, context_gate=context_gate)
        self.mtc.embeddings_2 = TinyEmbedding(3, context_gate=context_gate)


def validation_metrics(epoch: int, val_loss: float) -> dict:
    return {
        "val_loss": val_loss,
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


def make_runner(
    directory: Path,
    args: argparse.Namespace,
    components,
    *,
    spec: exact_runner.ExactRunSpec | None = None,
) -> entry.DCHExactRunner:
    model, optimizer, scaler, generator = components
    policy = selection_policy()
    return entry.DCHExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        spec=spec or tiny_spec(args, model, optimizer, scaler),
        selection_policy=policy,
        compatibility_payload_factory=entry.EvaluatorCheckpointAdapter(
            model_metadata={"variant": "tpd_clean_v7_dch_full"},
            split_hashes={"train": "a" * 64},
        ),
    )


def run_one_epoch(runner: exact_runner.ExactRunner, components) -> dict:
    model, optimizer, scaler, generator = components
    control = runner.next_epoch_control()
    permutation = torch.randperm(9, generator=generator)
    python_value = random.getrandbits(31)
    numpy_value = int(np.random.randint(0, 2**31 - 1))
    inputs = torch.rand(2, 3)
    targets = torch.tensor(
        [
            [(python_value % 101) / 101.0, (numpy_value % 103) / 103.0],
            [(numpy_value % 107) / 107.0, (python_value % 109) / 109.0],
        ]
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    scaler.update()
    trace = {
        "permutation": permutation.tolist(),
        "python": python_value,
        "numpy": numpy_value,
        "loss": float(loss.detach()),
    }
    runner.commit_epoch(
        {
            "variant": "tpd_clean_v7_dch_full",
            "train_loss": trace["loss"],
            "processed_train_samples": 2,
            "epoch_seconds": 0.125,
            "loader_trace": trace["permutation"],
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            **validation_metrics(control.epoch, trace["loss"]),
        },
        extra_state={"formal_eps": entry.FORMAL_EPS},
    )
    return {
        "epoch": control.epoch,
        "learning_rate": control.learning_rate,
        **trace,
    }


class ExactCleanV7DCHEntryTests(unittest.TestCase):
    def test_import_does_not_rebind_shared_training_runner(self) -> None:
        from experiments import train_tpd_clean_v7_dch as thin

        self.assertIsNot(base.build_model, thin.build_clean_v7_dch_model)
        self.assertNotEqual(
            tuple(base.SUPPORTED_VARIANTS),
            SUPPORTED_CLEAN_V7_DCH_VARIANTS,
        )

    def test_cli_owns_dch_variants_paths_and_formal_axes(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            args = entry.parse_args(
                [
                    "--variant",
                    variant,
                    "--device",
                    "cpu",
                    "--allow-cpu-smoke",
                    "--fresh",
                ]
            )
            self.assertEqual(args.epochs, 800)
            self.assertEqual(args.eval_every, 1)
            self.assertEqual(args.workers, 0)
            self.assertIs(args.amp, False)
            self.assertEqual(args.eps, 1e-6)
            self.assertEqual(args.output_root, entry.DEFAULT_OUTPUT_ROOT)
            self.assertEqual(
                args.exact_source_lock,
                entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
            )
            self.assertIn("tpd_clean_v7_dch", str(args.output_root))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(
                [
                    "--variant",
                    "tpd_clean_v6_full",
                    "--device",
                    "cpu",
                    "--allow-cpu-smoke",
                    "--fresh",
                ]
            )

    def test_builder_is_directly_bound_to_dch_block_and_manifest(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            context_gate = 1.0 if variant.endswith("_full") else 0.0
            fake = TinyDCH(context_gate=context_gate)
            metadata = tiny_builder_metadata(variant)
            with mock.patch.object(
                entry,
                "build_clean_v7_dch_model",
                return_value=(fake, metadata),
            ) as builder:
                model, actual = entry.build_selected_model(variant, 73)
            builder.assert_called_once_with(variant, 73)
            self.assertIs(model, fake)
            manifest = actual["architecture_manifest"]
            self.assertEqual(
                manifest["schema"],
                entry.ARCHITECTURE_MANIFEST_SCHEMA,
            )
            self.assertEqual(
                manifest["block"],
                "model.tpd_clean_v7_dch.TPDCleanV7DCHBlock",
            )
            self.assertEqual(manifest["variant"], variant)
            self.assertEqual(
                manifest["embedding_topology"]["mtc.embeddings_1"][
                    "evidence_nodes"
                ],
                3,
            )
            self.assertEqual(
                manifest["embedding_topology"]["mtc.embeddings_2"][
                    "evidence_nodes"
                ],
                2,
            )
            self.assertTrue(
                all(
                    isinstance(block, TPDCleanV7DCHBlock)
                    and block.eps == entry.FORMAL_EPS
                    for block in (
                        *model.mtc.embeddings_1.blocks,
                        *model.mtc.embeddings_2.blocks,
                    )
                )
            )

    def test_runtime_sources_include_real_reused_v6_exact_dependencies(
        self,
    ) -> None:
        relative = {
            str(path.relative_to(entry.REPO_ROOT))
            for path in entry.RUNTIME_SOURCE_PATHS
        }
        self.assertEqual(
            len(entry.RUNTIME_SOURCE_PATHS),
            len(set(entry.RUNTIME_SOURCE_PATHS)),
        )
        self.assertTrue(all(path.is_file() for path in entry.RUNTIME_SOURCE_PATHS))
        self.assertTrue(
            {
                "experiments/train_tpd_clean_v7_dch_exact.py",
                "experiments/train_tpd_clean_v7_dch.py",
                "model/tpd_clean_v7_dch.py",
                "experiments/train_tpd_clean_v6_exact.py",
                "experiments/tpd_exact_runner.py",
                "experiments/tpd_exact_resume.py",
                "experiments/tpd_exact_epoch_journal.py",
                "experiments/tpd_exact_training_runtime.py",
                "experiments/train_tpd_pilot.py",
                "model/SCTransNet.py",
                "dataset.py",
                "utils.py",
                "warmup_scheduler.py",
                "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md",
            }
            <= relative
        )
        self.assertFalse(
            any(
                token in path
                for path in relative
                for token in (
                    "evaluate_tpd_clean_v6",
                    "smoke_tpd_clean_v6",
                    "launch_tpd_clean_v6",
                    "test_train_tpd_clean_v6",
                )
            )
        )

    def test_temporary_source_lock_uses_only_dch_lock_identity(self) -> None:
        training_digest = "a" * 64
        source_hashes = {
            str(path.relative_to(entry.REPO_ROOT)): entry.file_sha256(path)
            for path in entry.RUNTIME_SOURCE_PATHS
        }
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "temporary-dch-exact-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "variants": list(SUPPORTED_CLEAN_V7_DCH_VARIANTS),
                "formal_contract": entry.formal_contract(),
                "training_data_sha256": training_digest,
                "source_sha256": source_hashes,
            }
            lock_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            contract = entry.source_lock_contract(
                training_digest,
                lock_path,
            )
            self.assertIn("tpd_clean_v7_dch_exact_source_lock", contract)
            self.assertNotIn("tpd_clean_v6_exact_source_lock", contract)
            self.assertEqual(contract["training_data"], training_digest)

            payload["schema"] = (
                "sctransnet_tpd_clean_v6_exact_source_lock_v1"
            )
            lock_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                entry.source_lock_contract(training_digest, lock_path)

    def test_run_spec_has_dch_identity_and_keeps_five_metric_order(self) -> None:
        with cpu_rng_environment():
            args = tiny_args()
            model, optimizer, scaler, _ = tiny_components()
            spec = tiny_spec(args, model, optimizer, scaler)
        normalized = spec.normalized()
        self.assertTrue(
            normalized["run_id"].startswith("tpd-clean-v7-dch-exact:")
        )
        self.assertNotIn("tpd-clean-v6-exact:", normalized["run_id"])
        self.assertEqual(
            normalized["determinism"]["entry_schema"],
            entry.ENTRY_SCHEMA,
        )
        primary = normalized["selection_policy"]["primary"]
        secondary = normalized["selection_policy"]["secondary"]
        self.assertEqual(
            [item["name"] for item in primary["order"]],
            ["pd", "fa", "tiny_pd", "miou", "val_loss"],
        )
        self.assertEqual(
            [item["name"] for item in secondary["order"]],
            ["miou", "pd", "fa", "tiny_pd", "val_loss"],
        )
        self.assertEqual(
            primary["stored_metrics"],
            list(entry.STORED_VALIDATION_METRICS),
        )
        self.assertEqual(
            secondary["stored_metrics"],
            list(entry.STORED_VALIDATION_METRICS),
        )

    def test_checkpoint_adapter_requires_and_stores_all_17_metrics(self) -> None:
        metrics = validation_metrics(3, 0.25)
        adapter = entry.EvaluatorCheckpointAdapter(
            model_metadata={"variant": "tpd_clean_v7_dch_full"},
            split_hashes={"train": "a" * 64},
        )
        exact_payload = {
            "model": {"state_dict": {"weight": torch.tensor([1.0])}},
            "optimizer": {"state_dict": {"state": {"step": 3}}},
            "scaler": {"state_dict": {"scale": 1.0}},
        }
        context = exact_runner.CompatibilityPayloadContext(
            role="last_evaluated_epoch",
            epoch=3,
            metrics=metrics,
            event={"epoch": 3, **metrics},
            run_identity={
                "variant": "tpd_clean_v7_dch_full",
                "dataset": "NUDT-SIRST",
                "seed": 42,
                "split_seed": 20260722,
            },
            exact_payload=exact_payload,
            normalized_spec={"total_epochs": 800},
        )
        payload = adapter(context)
        self.assertEqual(
            tuple(payload["validation_metrics"]),
            entry.STORED_VALIDATION_METRICS,
        )
        self.assertTrue(
            nested_equal(
                payload["state_dict"],
                exact_payload["model"]["state_dict"],
            )
        )
        incomplete = copy.deepcopy(context)
        del incomplete.metrics["niou"]
        with self.assertRaisesRegex(ValueError, "niou"):
            adapter(incomplete)

    def test_completion_summary_preserves_all_17_fields(self) -> None:
        args = tiny_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = []
            for epoch in (1, 2):
                events.append(
                    {
                        "epoch": epoch,
                        "epoch_seconds": 0.1,
                        "skipped_singleton_batches": 0,
                        **validation_metrics(epoch, 0.3 / epoch),
                    }
                )
            (root / exact_runner.METRICS_FILENAME).write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            with mock.patch.object(entry, "FORMAL_EPOCHS", 2):
                summary = entry.completion_summary(
                    args,
                    directory=root,
                    model_metadata={
                        "variant": "tpd_clean_v7_dch_full"
                    },
                    split_hashes={"train": "a" * 64},
                    selection={
                        "primary": {"epoch": 1},
                        "secondary": {"epoch": 2},
                    },
                )
        self.assertEqual(
            tuple(summary["best_pd_validation_metrics"]),
            entry.STORED_VALIDATION_METRICS,
        )
        self.assertEqual(
            tuple(summary["best_miou_validation_metrics"]),
            entry.STORED_VALIDATION_METRICS,
        )
        self.assertEqual(
            summary["schema"],
            "sctransnet_tpd_clean_v7_dch_completion_summary_v1",
        )

    def test_one_epoch_checkpoint_and_exact_resume_keep_dch_contract(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()
            first_components = tiny_components()
            first_spec = tiny_spec(
                args,
                first_components[0],
                first_components[1],
                first_components[2],
            )
            first = make_runner(
                root,
                args,
                first_components,
                spec=first_spec,
            )
            first.startup(exact_runner.InitializationRequest.fresh())
            run_one_epoch(first, first_components)

            last_payload = torch.load(
                root / exact_runner.LAST_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                last_payload["checkpoint_role"],
                "last_evaluated_epoch",
            )
            self.assertEqual(
                tuple(last_payload["validation_metrics"]),
                entry.STORED_VALIDATION_METRICS,
            )
            self.assertTrue(
                last_payload["run_identity"]["run_id"].startswith(
                    "tpd-clean-v7-dch-exact:"
                )
            )

            rebuilt_components = tiny_components(seed=999)
            rebuilt_spec = tiny_spec(
                args,
                rebuilt_components[0],
                rebuilt_components[1],
                rebuilt_components[2],
                initial_model_state_sha256=(
                    first_spec.initial_model_state_sha256
                ),
                initial_rng=first_spec.initial_rng,
                policy=first_spec.selection_policy,
            )
            rebuilt = make_runner(
                root,
                args,
                rebuilt_components,
                spec=rebuilt_spec,
            )
            snapshot = rebuilt.startup(
                exact_runner.InitializationRequest.exact()
            )
            self.assertEqual(snapshot.completed_epoch, 1)
            self.assertEqual(snapshot.next_epoch, 2)

    def test_exact_resume_plan_rejects_v6_protocol_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": "sctransnet_tpd_clean_v6_exact_entry_v1",
                        "run_identity": {},
                    }
                ),
                encoding="utf-8",
            )
            args = parse(["--exact-resume"])
            model = tiny_components()[0]
            with self.assertRaisesRegex(ValueError, "not a V7-DCH"):
                entry.initialization_plan(args, root, model)

    def test_cpu_device_and_six_output_objective(self) -> None:
        args = tiny_args()
        args.allow_cpu_smoke = False
        with self.assertRaisesRegex(ValueError, "allow-cpu-smoke"):
            entry.resolve_device(args)
        args.allow_cpu_smoke = True
        with mock.patch.object(
            torch.cuda,
            "is_available",
            side_effect=AssertionError("GPU query is not allowed for CPU"),
        ):
            self.assertEqual(entry.resolve_device(args), torch.device("cpu"))

        target = torch.zeros(2, 1, 4, 4)
        outputs = tuple(
            torch.full_like(target, 0.25 + index * 0.05)
            for index in range(6)
        )
        criterion = nn.BCELoss(reduction="mean")
        actual = entry.six_output_bce_loss(outputs, target, criterion)
        expected = base.deep_supervision_loss(outputs, target, criterion)
        torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(RuntimeError, "exactly six"):
            entry.six_output_bce_loss(outputs[:5], target, criterion)

    def test_environment_contract_records_verified_physical_gpu(self) -> None:
        expected_uuid = entry.PHYSICAL_GPU_UUIDS["2"]
        shared_payload = {
            "device_type": "cuda",
            "logical_device": "cuda:0",
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
                entry.os.environ,
                {
                    "TPD_DCH_PHYSICAL_GPU_INDEX": "2",
                    "TPD_DCH_PHYSICAL_GPU_UUID": expected_uuid,
                },
                clear=False,
            ),
        ):
            actual = entry.environment_contract(torch.device("cuda:0"))
        self.assertEqual(actual["physical_gpu_index"], 2)
        self.assertEqual(actual["physical_gpu_uuid"], expected_uuid)
        self.assertEqual(
            actual["physical_gpu_assignment_source"],
            "verified_worker_environment",
        )

        with (
            mock.patch.object(
                entry.shared_exact,
                "environment_contract",
                return_value=copy.deepcopy(shared_payload),
            ),
            mock.patch.dict(
                entry.os.environ,
                {
                    "TPD_DCH_PHYSICAL_GPU_INDEX": "1",
                    "TPD_DCH_PHYSICAL_GPU_UUID": expected_uuid,
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU 2 or 3"):
                entry.environment_contract(torch.device("cuda:0"))

        with mock.patch.object(
            entry.shared_exact,
            "environment_contract",
            return_value={"device_type": "cpu", "logical_device": "cpu"},
        ):
            cpu = entry.environment_contract(torch.device("cpu"))
        self.assertIsNone(cpu["physical_gpu_index"])
        self.assertIsNone(cpu["physical_gpu_uuid"])


if __name__ == "__main__":
    unittest.main()
