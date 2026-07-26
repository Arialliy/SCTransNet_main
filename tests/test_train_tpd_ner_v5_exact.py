from __future__ import annotations

import argparse
import copy
import hashlib
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

from experiments import tpd_exact_runner as exact_runner
from experiments import train_tpd_ner_v5_exact as entry
from experiments import train_tpd_pilot as base
from experiments.train_tpd_ner_v5 import (
    SUPPORTED_TPD_NER_V5_VARIANTS,
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


def parse(arguments: list[str]) -> object:
    return entry.parse_args(
        [
            "--variant",
            "tpd_clean_v5_full_relay_off",
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--epochs",
            "2",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "2",
            "--patch-size",
            "32",
            *arguments,
        ]
    )


def tiny_components(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return model, optimizer, scaler, generator


def tiny_args() -> argparse.Namespace:
    return entry.parse_args(
        [
            "--variant",
            "tpd_clean_v5_full_relay_off",
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--epochs",
            "2",
            "--warmup-epochs",
            "1",
            "--batch-size",
            "2",
            "--patch-size",
            "32",
            "--base-lr",
            "0.004",
            "--min-lr",
            "0.0001",
            "--fresh",
        ]
    )


def tiny_metadata() -> dict:
    return {
        "variant": "tpd_clean_v5_full_relay_off",
        "candidate_family": "fixture",
        "architecture_manifest": {
            "model": "tiny",
            "width": 4,
            "outputs": 2,
        },
    }


def tiny_spec(
    args,
    model,
    optimizer,
    scaler,
    *,
    train_order=("a", "b"),
    normalization='{"mean":1.0,"std":2.0}',
    source_digest="1" * 64,
    environment_name="cpu-fixture",
    metadata=None,
    initial_model_state_sha256=None,
    initial_rng=None,
    selection_policy=None,
):
    split_records = {
        "train": exact_runner.OrderedFingerprint.from_values(
            "train",
            train_order,
        ),
        "validation": exact_runner.OrderedFingerprint.from_values(
            "validation",
            ("c",),
        ),
    }
    data_records = {
        "train_samples": exact_runner.OrderedFingerprint.from_values(
            "train_samples",
            tuple(f"{item}:content" for item in train_order),
        ),
        "normalization": exact_runner.OrderedFingerprint.from_values(
            "normalization",
            (normalization,),
        ),
    }
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=metadata or tiny_metadata(),
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
        source_locks={"fixture": source_digest},
        split_records=split_records,
        data_records=data_records,
        environment={"name": environment_name},
    )


class ExactV5NEREntryTests(unittest.TestCase):
    def test_cli_initialization_modes_are_mutually_exclusive(self) -> None:
        fresh = parse(["--fresh"])
        self.assertTrue(fresh.fresh)
        self.assertEqual(
            fresh.exact_source_lock,
            entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
        )
        self.assertTrue(parse(["--exact-resume"]).exact_resume)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.pth.tar"
            parent.write_bytes(b"parent")
            identity = root / "identity.json"
            identity.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(b"parent").hexdigest()
            args = parse(
                [
                    "--same-layout-parent",
                    str(parent),
                    "--parent-checkpoint-sha256",
                    digest,
                    "--parent-identity-json",
                    str(identity),
                    "--parent-epoch",
                    "7",
                ]
            )
            self.assertEqual(args.same_layout_parent, parent)

        for invalid in (
            [],
            ["--fresh", "--exact-resume"],
            ["--same-layout-parent", "parent.pth.tar"],
            ["--fresh", "--parent-epoch", "1"],
            ["--fresh", "--workers", "1"],
            ["--fresh", "--eval-every", "2"],
        ):
            with self.subTest(arguments=invalid):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parse(invalid)

    def test_direct_builder_routes_all_four_variants_without_monkeypatch(self) -> None:
        original_builder = base.build_model
        sentinel_model = nn.Linear(1, 1)
        with mock.patch.object(
            entry,
            "build_tpd_ner_v5_model",
            return_value=(sentinel_model, {"architecture_manifest": {"x": 1}}),
        ) as builder:
            for variant in SUPPORTED_TPD_NER_V5_VARIANTS:
                with self.subTest(variant=variant):
                    model, _ = entry.build_selected_model(
                        variant,
                        73,
                        img_size=32,
                        relay_width=2,
                    )
                    self.assertIs(model, sentinel_model)
                    builder.assert_called_with(
                        variant,
                        73,
                        img_size=32,
                        relay_width=2,
                    )
        self.assertIs(base.build_model, original_builder)

    def test_run_identity_changes_with_every_entry_owned_contract_axis(self) -> None:
        args = tiny_args()
        model, optimizer, scaler, _ = tiny_components()
        baseline_spec = tiny_spec(args, model, optimizer, scaler)
        baseline = exact_runner.build_run_identity(model, baseline_spec)
        self.assertEqual(
            baseline_spec.optimizer["param_groups"][0]["parameter_names"],
            [name for name, _ in model.named_parameters()],
        )

        cases = {}
        changed_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            weight_decay=0.1,
        )
        cases["optimizer"] = tiny_spec(
            args,
            model,
            changed_optimizer,
            scaler,
        )
        cases["train_order"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            train_order=("b", "a"),
        )
        cases["normalization"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            normalization='{"mean":1.5,"std":2.0}',
        )
        cases["source"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            source_digest="2" * 64,
        )
        cases["environment"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            environment_name="different-cpu",
        )
        changed_metadata = tiny_metadata()
        changed_metadata["architecture_manifest"]["width"] = 5
        cases["builder_manifest"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            metadata=changed_metadata,
        )
        changed_args = copy.deepcopy(args)
        changed_args.threshold = 0.6
        cases["hyperparameter"] = tiny_spec(
            changed_args,
            model,
            optimizer,
            scaler,
        )
        changed_rng = copy.deepcopy(baseline_spec.initial_rng)
        changed_rng["torch_cpu_sha256"] = "e" * 64
        cases["initial_rng"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            initial_rng=changed_rng,
        )
        changed_selection = copy.deepcopy(baseline_spec.selection_policy)
        changed_selection["primary"]["order"][0]["maximize"] = False
        cases["selection_policy"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            selection_policy=changed_selection,
        )
        cases["initial_model_state"] = tiny_spec(
            args,
            model,
            optimizer,
            scaler,
            initial_model_state_sha256="f" * 64,
        )
        for label, spec in cases.items():
            with self.subTest(label=label):
                changed = exact_runner.build_run_identity(model, spec)
                self.assertNotEqual(
                    baseline["contract_sha256"],
                    changed["contract_sha256"],
                )

    def test_exact_resume_plan_reuses_original_initial_identity_fields(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()
            model, optimizer, scaler, _ = tiny_components()
            original_spec = tiny_spec(args, model, optimizer, scaler)
            original_identity = exact_runner.build_run_identity(
                model,
                original_spec,
            )
            (root / "protocol.json").write_text(
                json.dumps({"run_identity": original_identity}),
                encoding="utf-8",
            )

            exact_args = parse(["--exact-resume"])
            rebuilt_model, _, _, _ = tiny_components(seed=999)
            plan = entry.initialization_plan(
                exact_args,
                root,
                rebuilt_model,
            )
            training = original_identity["training_contract"]
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

    def test_exact_source_lock_recursively_binds_the_frozen_ner_lock(
        self,
    ) -> None:
        ner_lock = json.loads(
            entry.SOURCE_LOCK_PATH.read_text(encoding="utf-8")
        )
        training_digest = ner_lock["training_data_sha256"]
        runtime_sources = (
            Path(entry.__file__).resolve(),
            *entry.EXACT_CORE_PATHS,
        )
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "exact-lock.json"
            payload = {
                "schema": entry.EXACT_SOURCE_LOCK_SCHEMA,
                "recursive_source_locks": {
                    "experiments/tpd_ner_v5_source_lock.json": (
                        entry.file_sha256(entry.SOURCE_LOCK_PATH)
                    )
                },
                "source_sha256": {
                    str(path.relative_to(entry.REPO_ROOT)): (
                        entry.file_sha256(path)
                    )
                    for path in runtime_sources
                },
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
                contract["tpd_ner_v5_exact_source_lock"],
                entry.file_sha256(lock_path),
            )
            self.assertEqual(
                contract["tpd_ner_v5_recursive_source_lock"],
                entry.file_sha256(entry.SOURCE_LOCK_PATH),
            )

            payload["recursive_source_locks"][
                "experiments/tpd_ner_v5_source_lock.json"
            ] = "0" * 64
            lock_path.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not bind"):
                entry.source_lock_contract(training_digest, lock_path)

    def test_cpu_smoke_is_explicit_and_gpu_runtime_contract_is_checked(
        self,
    ) -> None:
        cpu_args = tiny_args()
        cpu_args.allow_cpu_smoke = False
        with self.assertRaisesRegex(ValueError, "allow-cpu-smoke"):
            entry.resolve_device(cpu_args)

        gpu_args = copy.deepcopy(cpu_args)
        gpu_args.device = "cuda:0"
        gpu_args.allow_cpu_smoke = False
        properties = SimpleNamespace(uuid="GPU-fixture")
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
                {"CUDA_VISIBLE_DEVICES": "GPU-fixture"},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
                entry.resolve_device(gpu_args)
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
                    "PYTHONHASHSEED": str(gpu_args.seed),
                },
                clear=True,
            ),
        ):
            self.assertEqual(
                entry.resolve_device(gpu_args),
                torch.device("cuda:0"),
            )

    def test_exact_resume_starts_at_completed_plus_one_and_adapter_is_legacy_usable(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tiny_args()
            metadata = tiny_metadata()
            split_hashes = {
                "used_train_sha256": "a" * 64,
                "used_val_sha256": "b" * 64,
            }
            adapter = entry.EvaluatorCheckpointAdapter(
                model_metadata=metadata,
                split_hashes=split_hashes,
            )

            model, optimizer, scaler, generator = tiny_components()
            spec = tiny_spec(args, model, optimizer, scaler)
            writer = exact_runner.ExactRunner(
                root,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader_generator=generator,
                spec=spec,
                compatibility_payload_factory=adapter,
            )
            writer.startup(exact_runner.InitializationRequest.fresh())
            control = writer.next_epoch_control()
            self.assertEqual(control.epoch, 1)
            inputs = torch.randn(2, 3)
            target = torch.randn(2, 2)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs) - target).square().mean()
            loss.backward()
            optimizer.step()
            writer.commit_epoch(
                {
                    "variant": args.variant,
                    "train_loss": float(loss.detach()),
                    "processed_train_samples": 2,
                    "epoch_seconds": 0.1,
                    "pd": 0.9,
                    "fa": 1.0e-6,
                    "tiny_pd": 1.0,
                    "miou": 0.8,
                    "val_loss": 0.2,
                }
            )
            checkpoint = torch.load(
                root / exact_runner.LAST_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            required = {
                "state_dict",
                "optimizer",
                "scaler",
                "validation_metrics",
                "model_metadata",
                "split_hashes",
            }
            self.assertTrue(required <= set(checkpoint))
            self.assertEqual(
                checkpoint["validation_metrics"],
                {
                    "pd": 0.9,
                    "fa": 1.0e-6,
                    "tiny_pd": 1.0,
                    "miou": 0.8,
                    "val_loss": 0.2,
                },
            )
            self.assertEqual(checkpoint["model_metadata"], metadata)
            self.assertEqual(checkpoint["split_hashes"], split_hashes)

            rebuilt_model, rebuilt_optimizer, rebuilt_scaler, rebuilt_generator = (
                tiny_components(seed=999)
            )
            rebuilt_generator.manual_seed(args.seed)
            rebuilt_spec = tiny_spec(
                args,
                rebuilt_model,
                rebuilt_optimizer,
                rebuilt_scaler,
                initial_model_state_sha256=(
                    spec.initial_model_state_sha256
                ),
                initial_rng=spec.initial_rng,
                selection_policy=spec.selection_policy,
            )
            reader = exact_runner.ExactRunner(
                root,
                model=rebuilt_model,
                optimizer=rebuilt_optimizer,
                scaler=rebuilt_scaler,
                loader_generator=rebuilt_generator,
                spec=rebuilt_spec,
                compatibility_payload_factory=adapter,
            )
            snapshot = reader.startup(
                exact_runner.InitializationRequest.exact()
            )
            self.assertEqual(snapshot.completed_epoch, 1)
            self.assertEqual(snapshot.next_epoch, 2)
            self.assertEqual(reader.next_epoch_control().epoch, 2)

    def test_six_output_loss_keeps_the_baseline_objective(self) -> None:
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

    def test_legacy_entry_remains_byte_identical_to_its_frozen_lock(self) -> None:
        lock = json.loads(
            entry.SOURCE_LOCK_PATH.read_text(encoding="utf-8")
        )
        expected = lock["source_sha256"]["experiments/train_tpd_ner_v5.py"]
        actual = entry.file_sha256(
            entry.REPO_ROOT / "experiments/train_tpd_ner_v5.py"
        )
        self.assertEqual(actual, expected)
        source = (
            entry.REPO_ROOT / "experiments/train_tpd_ner_v5.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("train_tpd_ner_v5_exact", source)


if __name__ == "__main__":
    unittest.main()
