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
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact_resume
from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_clean_v6_exact as entry
from experiments import train_tpd_pilot as base
from model.tpd_clean_v6 import (
    SUPPORTED_CLEAN_V6_VARIANTS,
    TPDCleanV6Block,
)


torch.set_num_threads(1)


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
            "tpd_clean_v6_full",
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
        "variant": "tpd_clean_v6_full",
        "candidate_family": "fixture",
        "architecture_manifest": {
            "schema": entry.ARCHITECTURE_MANIFEST_SCHEMA,
            "model": "tiny",
            "width": 5,
            "outputs": 2,
            "eps": entry.FORMAL_EPS,
        },
    }


def tiny_spec(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    initial_model_state_sha256: str | None = None,
    initial_rng: dict | None = None,
    selection_policy: dict | None = None,
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
            copy.deepcopy(selection_policy)
            if selection_policy is not None
            else exact_runner.pd_miou_selection_policy().normalized()
        ),
        source_locks={"fixture": "1" * 64},
        split_records=split_records,
        data_records=data_records,
        environment={"name": "cpu-fixture"},
    )


def tiny_builder_metadata(variant: str) -> dict:
    return {
        "variant": variant,
        "candidate_family": "fixture-v6",
        "mainline_contract": "Keep-Context-Saliency",
        "semantic_sources": ("Keep", "Context", "Saliency"),
        "replaced_embeddings": ("mtc.embeddings_1", "mtc.embeddings_2"),
        "phase_tied_projection_formula": "Wt=sum_phase(Wk)",
        "context_code_formula": "Q=tanh(centered/rms_eps)",
        "context_headroom_formula": "H=1+...",
        "fusion_equation": "K+Sa*(a*H)",
        "derived_projection_parameters": 0,
        "derived_projection_buffers": 0,
        "shallow_embedding_parameters": 66_176,
        "total_parameters": 10_843_155,
    }


class TinyEmbedding(nn.Module):
    def __init__(self, count: int, *, use_context_headroom: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TPDCleanV6Block(
                1,
                activate=index < count - 1,
                use_context_headroom=use_context_headroom,
            )
            for index in range(count)
        )


class TinyV6(nn.Module):
    def __init__(self, *, use_context_headroom: bool) -> None:
        super().__init__()
        self.mtc = nn.Module()
        self.mtc.embeddings_1 = TinyEmbedding(
            4,
            use_context_headroom=use_context_headroom,
        )
        self.mtc.embeddings_2 = TinyEmbedding(
            3,
            use_context_headroom=use_context_headroom,
        )


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
) -> entry.V6ExactRunner:
    model, optimizer, scaler, generator = components
    return entry.V6ExactRunner(
        directory,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        spec=spec or tiny_spec(args, model, optimizer, scaler),
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
            "variant": "tpd_clean_v6_full",
            "train_loss": trace["loss"],
            "processed_train_samples": 2,
            "epoch_seconds": 0.125,
            "loader_trace": trace["permutation"],
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            "pd": 0.70 + 0.01 * control.epoch,
            "fa": 1.0e-5 / control.epoch,
            "tiny_pd": 0.60 + 0.01 * control.epoch,
            "miou": 0.50 + 0.02 * control.epoch,
            "val_loss": trace["loss"],
        },
        extra_state={"formal_eps": entry.FORMAL_EPS},
    )
    return {
        "epoch": control.epoch,
        "learning_rate": control.learning_rate,
        **trace,
    }


class ExactCleanV6EntryTests(unittest.TestCase):
    def test_cli_exposes_only_fresh_or_exact_resume_and_forces_formal_axes(
        self,
    ) -> None:
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            with self.subTest(variant=variant):
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
        self.assertTrue(parse(["--exact-resume"]).exact_resume)

        invalid = (
            [],
            ["--fresh", "--exact-resume"],
            ["--fresh", "--epochs", "799"],
            ["--fresh", "--eval-every", "2"],
            ["--fresh", "--workers", "1"],
            ["--fresh", "--eps", "1e-5"],
            ["--fresh", "--amp"],
            ["--same-layout-parent", "parent.pth.tar"],
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parse(list(arguments))

    def test_builder_is_directly_bound_to_both_v6_variants_and_exact_eps(
        self,
    ) -> None:
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            with self.subTest(variant=variant):
                fake = TinyV6(
                    use_context_headroom=variant.endswith("_full")
                )
                metadata = tiny_builder_metadata(variant)
                with mock.patch.object(
                    entry,
                    "build_clean_v6_model",
                    return_value=(fake, metadata),
                ) as builder:
                    model, actual = entry.build_selected_model(
                        variant,
                        73,
                    )
                builder.assert_called_once_with(variant, 73)
                self.assertIs(model, fake)
                self.assertEqual(actual["formal_eps"], 1e-6)
                self.assertIs(actual["formal_amp"], False)
                manifest = actual["architecture_manifest"]
                self.assertEqual(
                    manifest["schema"],
                    entry.ARCHITECTURE_MANIFEST_SCHEMA,
                )
                self.assertEqual(manifest["variant"], variant)
                self.assertEqual(manifest["eps"], 1e-6)
                self.assertIs(manifest["formal_amp"], False)
                blocks = [
                    *model.mtc.embeddings_1.blocks,
                    *model.mtc.embeddings_2.blocks,
                ]
                self.assertTrue(
                    all(
                        isinstance(block, TPDCleanV6Block)
                        and block.eps == 1e-6
                        for block in blocks
                    )
                )

        with self.assertRaisesRegex(ValueError, "eps"):
            entry.build_selected_model(
                "tpd_clean_v6_full",
                42,
                eps=1e-5,
            )

    def test_temporary_source_lock_binds_every_v6_exact_runtime_source(
        self,
    ) -> None:
        self.assertFalse(entry.DEFAULT_EXACT_SOURCE_LOCK_PATH.exists())
        training_digest = "a" * 64
        relative_paths = {
            str(path.relative_to(entry.REPO_ROOT))
            for path in entry.RUNTIME_SOURCE_PATHS
        }
        self.assertTrue(
            {
                "experiments/train_tpd_clean_v6_exact.py",
                "experiments/train_tpd_clean_v6.py",
                "model/tpd_clean_v6.py",
                "experiments/evaluate_tpd_clean_v6_pd_fa.py",
                "experiments/evaluate_pd_fa_sweep.py",
                "experiments/smoke_tpd_clean_v6.py",
                "experiments/TPD_CLEAN_V6_PROTOCOL.md",
                "experiments/tpd_exact_runner.py",
                "experiments/tpd_exact_resume.py",
                "experiments/tpd_exact_epoch_journal.py",
                "experiments/tpd_exact_training_runtime.py",
                "experiments/tpd_extension_warm_start.py",
                "experiments/train_tpd_pilot.py",
                "model/SCTransNet.py",
                "model/Config.py",
                "model/tpd.py",
                "dataset.py",
                "utils.py",
                "warmup_scheduler.py",
                "tests/test_train_tpd_clean_v6_exact.py",
            }
            <= relative_paths
        )
        source_hashes = {
            str(path.relative_to(entry.REPO_ROOT)): entry.file_sha256(path)
            for path in entry.RUNTIME_SOURCE_PATHS
        }
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "temporary-v6-exact-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "variants": list(SUPPORTED_CLEAN_V6_VARIANTS),
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
            self.assertEqual(
                contract["tpd_clean_v6_exact_source_lock"],
                entry.file_sha256(lock_path),
            )
            self.assertEqual(contract["training_data"], training_digest)
            for relative, digest in source_hashes.items():
                self.assertEqual(
                    contract[f"exact_source:{relative}"],
                    digest,
                )

            evaluator = "experiments/evaluate_tpd_clean_v6_pd_fa.py"
            payload["source_sha256"][evaluator] = "0" * 64
            lock_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                entry.source_lock_contract(training_digest, lock_path)

            payload["source_sha256"] = dict(source_hashes)
            del payload["source_sha256"][evaluator]
            lock_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "omits"):
                entry.source_lock_contract(training_digest, lock_path)

    def test_run_spec_binds_formal_contract_and_optimizer_parameter_order(
        self,
    ) -> None:
        with cpu_rng_environment():
            args = tiny_args()
            model, optimizer, scaler, _ = tiny_components()
            spec = tiny_spec(args, model, optimizer, scaler)
            normalized = spec.normalized()
            self.assertEqual(normalized["total_epochs"], 800)
            self.assertEqual(normalized["eval_interval"], 1)
            self.assertEqual(normalized["workers"], 0)
            self.assertIs(normalized["amp"], False)
            self.assertEqual(
                normalized["determinism"]["eps"],
                entry.FORMAL_EPS,
            )
            self.assertEqual(
                normalized["determinism"]["formal_contract"],
                entry.formal_contract(),
            )
            self.assertEqual(
                normalized["optimizer"]["param_groups"][0][
                    "parameter_names"
                ],
                [name for name, _ in model.named_parameters()],
            )
            self.assertEqual(
                normalized["deep_supervision"]["expected_outputs"],
                6,
            )

    def test_exact_resume_plan_reuses_original_initial_identity_fields(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()
            model, optimizer, scaler, _ = tiny_components()
            original_spec = tiny_spec(args, model, optimizer, scaler)
            identity = exact_runner.build_run_identity(model, original_spec)
            (root / "protocol.json").write_text(
                json.dumps({"run_identity": identity}),
                encoding="utf-8",
            )
            rebuilt_model, _, _, _ = tiny_components(seed=999)
            plan = entry.initialization_plan(
                parse(["--exact-resume"]),
                root,
                rebuilt_model,
            )
            training = identity["training_contract"]
            self.assertEqual(
                plan.initial_model_state_sha256,
                training["initial_model_state_sha256"],
            )
            self.assertEqual(plan.initial_rng, training["initial_rng"])
            self.assertEqual(
                plan.selection_policy,
                training["selection_policy"],
            )
            self.assertEqual(
                plan.contract,
                training["initialization_contract"],
            )
            self.assertNotEqual(
                plan.initial_model_state_sha256,
                exact_runner.initial_model_state_sha256(rebuilt_model),
            )

    def test_three_epoch_continuous_and_one_plus_exact_resume_are_identical(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()

            continuous_components = tiny_components()
            continuous = make_runner(
                root / "continuous",
                args,
                continuous_components,
            )
            continuous.startup(exact_runner.InitializationRequest.fresh())
            continuous_trace = [
                run_one_epoch(continuous, continuous_components)
                for _ in range(3)
            ]
            continuous_model = copy.deepcopy(
                continuous_components[0].state_dict()
            )
            continuous_optimizer = copy.deepcopy(
                continuous_components[1].state_dict()
            )
            continuous_rng = exact_resume.capture_rng_state(
                continuous_components[3]
            )
            continuous_metrics = (
                root / "continuous" / exact_runner.METRICS_FILENAME
            ).read_bytes()

            split_components = tiny_components()
            split_spec = tiny_spec(
                args,
                split_components[0],
                split_components[1],
                split_components[2],
            )
            split = make_runner(
                root / "split",
                args,
                split_components,
                spec=split_spec,
            )
            split.startup(exact_runner.InitializationRequest.fresh())
            split_trace = [run_one_epoch(split, split_components)]

            rebuilt_components = tiny_components(seed=999)
            rebuilt_spec = tiny_spec(
                args,
                rebuilt_components[0],
                rebuilt_components[1],
                rebuilt_components[2],
                initial_model_state_sha256=(
                    split_spec.initial_model_state_sha256
                ),
                initial_rng=split_spec.initial_rng,
                selection_policy=split_spec.selection_policy,
            )
            rebuilt = make_runner(
                root / "split",
                args,
                rebuilt_components,
                spec=rebuilt_spec,
            )
            snapshot = rebuilt.startup(
                exact_runner.InitializationRequest.exact()
            )
            self.assertEqual(snapshot.completed_epoch, 1)
            self.assertEqual(snapshot.next_epoch, 2)
            split_trace.extend(
                run_one_epoch(rebuilt, rebuilt_components)
                for _ in range(2)
            )
            rebuilt_rng = exact_resume.capture_rng_state(
                rebuilt_components[3]
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
                    rebuilt_components[0].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    continuous_optimizer,
                    rebuilt_components[1].state_dict(),
                )
            )
            self.assertTrue(nested_equal(continuous_rng, rebuilt_rng))
            last_payload = torch.load(
                root / "split" / exact_runner.LAST_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                last_payload["checkpoint_role"],
                "last_evaluated_epoch",
            )

    def test_adapter_exactly_copies_exact_model_optimizer_and_scaler(
        self,
    ) -> None:
        adapter = entry.EvaluatorCheckpointAdapter(
            model_metadata={"variant": "tpd_clean_v6_full"},
            split_hashes={"train": "a" * 64},
        )
        exact_payload = {
            "model": {"state_dict": {"weight": torch.tensor([1.0])}},
            "optimizer": {"state_dict": {"state": {"step": 3}}},
            "scaler": {"state_dict": {"scale": 1.0}},
        }
        context = exact_runner.CompatibilityPayloadContext(
            role="last",
            epoch=3,
            metrics={"pd": 0.9},
            event={"epoch": 3, "pd": 0.9},
            run_identity={
                "variant": "tpd_clean_v6_full",
                "dataset": "NUDT-SIRST",
                "seed": 42,
                "split_seed": 20260722,
            },
            exact_payload=exact_payload,
            normalized_spec={"total_epochs": 800},
        )
        payload = adapter(context)
        self.assertTrue(
            nested_equal(
                payload["state_dict"],
                exact_payload["model"]["state_dict"],
            )
        )
        self.assertTrue(
            nested_equal(
                payload["optimizer"],
                exact_payload["optimizer"]["state_dict"],
            )
        )
        self.assertTrue(
            nested_equal(
                payload["scaler"],
                exact_payload["scaler"]["state_dict"],
            )
        )
        payload["state_dict"]["weight"].add_(1.0)
        self.assertEqual(
            exact_payload["model"]["state_dict"]["weight"].item(),
            1.0,
        )

    def test_six_output_loss_keeps_existing_deep_supervision_objective(
        self,
    ) -> None:
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

    def test_cpu_execution_is_explicit_and_does_not_query_a_gpu(self) -> None:
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

    def test_cuda_contract_requires_deterministic_cublas_configuration(
        self,
    ) -> None:
        args = tiny_args()
        args.device = "cuda:0"
        args.allow_cpu_smoke = False
        properties = SimpleNamespace(uuid="fixture")
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
            mock.patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "GPU-fixture",
                    "PYTHONHASHSEED": str(args.seed),
                },
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CUBLAS_WORKSPACE_CONFIG",
            ):
                entry.resolve_device(args)

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
            mock.patch.dict(
                "os.environ",
                {
                    "CUDA_VISIBLE_DEVICES": "GPU-fixture",
                    "PYTHONHASHSEED": str(args.seed),
                    "CUBLAS_WORKSPACE_CONFIG": (
                        entry.FORMAL_CUBLAS_WORKSPACE_CONFIG
                    ),
                },
                clear=True,
            ),
        ):
            self.assertEqual(entry.resolve_device(args), torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
