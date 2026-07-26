from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments import tpd_exact_epoch_journal as journal_module
from experiments import tpd_exact_resume as exact
from experiments import tpd_exact_runner as runner_module


class StateScaler:
    def __init__(self, scale: float = 128.0, updates: int = 0) -> None:
        self.scale = float(scale)
        self.updates = int(updates)

    def state_dict(self) -> dict[str, float | int]:
        return {"scale": self.scale, "updates": self.updates}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.scale = float(state["scale"])
        self.updates = int(state["updates"])


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


def make_components(seed: int) -> dict:
    seed_all(seed)
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Tanh(),
        nn.Linear(5, 2),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.004,
        betas=(0.8, 0.95),
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(42)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": loader_generator,
    }


def make_spec(
    components: dict,
    *,
    batch_size: int = 2,
    workers: int = 0,
    manifest_sha256: str = "1" * 64,
    run_id: str = "runner-fixture",
    optimizer_declaration: dict | None = None,
    scaler_declaration: dict | None = None,
    initialization_contract: dict | None = None,
    initial_model_state_sha256: str | None = None,
    initial_rng: dict | None = None,
    selection_policy: dict | None = None,
) -> runner_module.ExactRunSpec:
    return runner_module.ExactRunSpec(
        run_id=run_id,
        variant="tpd_fixture",
        dataset="fixture",
        seed=42,
        split_seed=3407,
        builder_metadata={
            "builder": "tests.make_components",
            "width": 5,
            "output": 2,
        },
        builder_manifest_sha256=manifest_sha256,
        source_locks={
            "model": "2" * 64,
            "training": "3" * 64,
        },
        split_fingerprints={
            "train": runner_module.OrderedFingerprint.from_values(
                "train",
                ("a", "b", "c", "d"),
            ),
            "validation": runner_module.OrderedFingerprint.from_values(
                "validation",
                ("e", "f"),
            ),
        },
        data_fingerprints={
            "train_samples": runner_module.OrderedFingerprint.from_values(
                "train_samples",
                ("a:image0", "b:image1", "c:image2", "d:image3"),
            ),
            "validation_samples": runner_module.OrderedFingerprint.from_values(
                "validation_samples",
                ("e:image4", "f:image5"),
            ),
        },
        optimizer=(
            optimizer_declaration
            if optimizer_declaration is not None
            else runner_module.optimizer_contract(
                components["model"],
                components["optimizer"],
            )
        ),
        scaler=(
            scaler_declaration
            if scaler_declaration is not None
            else runner_module.scaler_contract(
                components["scaler"],
                amp=False,
            )
        ),
        initialization_contract=(
            initialization_contract
            if initialization_contract is not None
            else runner_module.fresh_initialization_contract()
        ),
        lr_schedule=runner_module.ManualCosineSchedule(
            total_epochs=4,
            base_lr=0.004,
            min_lr=0.0001,
            warmup_epochs=1,
        ),
        loss={"name": "mean_square", "reduction": "mean"},
        deep_supervision={"enabled": False, "outputs": 1},
        batch_size=batch_size,
        patch_size=32,
        workers=workers,
        amp=False,
        total_epochs=4,
        eval_interval=1,
        metric_config={
            "threshold": 0.5,
            "match_radius": 3.0,
            "tiny_area": 9,
        },
        environment={
            "torch": torch.__version__,
            "device_type": "cpu",
        },
        determinism={
            "workers": 0,
            "explicit_loader_generator": True,
            "manual_lr": True,
        },
        initial_model_state_sha256=(
            initial_model_state_sha256
            if initial_model_state_sha256 is not None
            else runner_module.initial_model_state_sha256(
                components["model"]
            )
        ),
        initial_rng=(
            copy.deepcopy(initial_rng)
            if initial_rng is not None
            else runner_module.initial_rng_contract()
        ),
        selection_policy=(
            copy.deepcopy(selection_policy)
            if selection_policy is not None
            else runner_module.pd_miou_selection_policy().normalized()
        ),
    )


def make_runner(
    root: Path,
    components: dict,
    *,
    spec: runner_module.ExactRunSpec | None = None,
    adapter=None,
    selection_policy: runner_module.SelectionPolicy | None = None,
) -> runner_module.ExactRunner:
    return runner_module.ExactRunner(
        root,
        model=components["model"],
        optimizer=components["optimizer"],
        scaler=components["scaler"],
        loader_generator=components["loader_generator"],
        spec=spec or make_spec(components),
        selection_policy=selection_policy,
        compatibility_payload_factory=adapter,
    )


def run_one_epoch(
    runner: runner_module.ExactRunner,
    components: dict,
) -> dict:
    control = runner.next_epoch_control()
    permutation = torch.randperm(
        7,
        generator=components["loader_generator"],
    )
    python_value = random.getrandbits(31)
    numpy_value = int(np.random.randint(0, 2**31 - 1))
    torch_values = torch.rand(6)
    inputs = torch_values.reshape(2, 3)
    target = torch.tensor(
        [
            [(python_value % 101) / 101.0, (numpy_value % 103) / 103.0],
            [(numpy_value % 107) / 107.0, (python_value % 109) / 109.0],
        ]
    )
    optimizer = components["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    loss = (components["model"](inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    components["scaler"].updates += 1
    metrics = {
        "pd": 0.70 + 0.01 * control.epoch,
        "fa": 1.0e-5 / control.epoch,
        "tiny_pd": 0.60 + 0.01 * control.epoch,
        "miou": 0.50 + 0.02 * control.epoch,
        "val_loss": float(loss.detach().item()),
    }
    runner.commit_epoch(
        {
            "train_loss": float(loss.detach().item()),
            "loader_trace": permutation.tolist(),
            "python_trace": python_value,
            "numpy_trace": numpy_value,
            **metrics,
        },
        extra_state={"scaler_updates": components["scaler"].updates},
    )
    return {
        "epoch": control.epoch,
        "learning_rate": control.learning_rate,
        "permutation": permutation.tolist(),
        "python": python_value,
        "numpy": numpy_value,
        "loss": float(loss.detach().item()),
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


class ExactRunnerTest(unittest.TestCase):
    def test_identity_binds_all_trajectory_fields_and_order(self) -> None:
        components = make_components(42)
        spec = make_spec(components)
        identity = runner_module.build_run_identity(
            components["model"],
            spec,
        )
        self.assertEqual(len(identity["architecture_id"]), 64)
        self.assertEqual(len(identity["contract_sha256"]), 64)
        self.assertEqual(identity["training_contract"]["workers"], 0)
        self.assertIsNone(
            identity["training_contract"]["manual_lr_schedule"]["scheduler"]
        )
        self.assertEqual(
            identity["builder_manifest_sha256"],
            "1" * 64,
        )
        reversed_split = copy.deepcopy(spec)
        reversed_split = runner_module.ExactRunSpec(
            **{
                **spec.__dict__,
                "split_fingerprints": {
                    **spec.split_fingerprints,
                    "train": runner_module.OrderedFingerprint.from_values(
                        "train",
                        ("d", "c", "b", "a"),
                    ),
                },
            }
        )
        changed = runner_module.build_run_identity(
            components["model"],
            reversed_split,
        )
        self.assertNotEqual(identity["split_sha256"], changed["split_sha256"])
        self.assertNotEqual(
            identity["contract_sha256"],
            changed["contract_sha256"],
        )

    def test_initial_model_and_global_rng_are_identity_bound(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            components = make_components(42)
            spec = make_spec(components)
            identity = runner_module.build_run_identity(
                components["model"],
                spec,
            )
            with torch.no_grad():
                next(components["model"].parameters()).add_(0.25)
            changed_spec = make_spec(
                components,
                initial_rng=spec.initial_rng,
                selection_policy=spec.selection_policy,
            )
            changed_identity = runner_module.build_run_identity(
                components["model"],
                changed_spec,
            )
            self.assertNotEqual(
                identity["contract_sha256"],
                changed_identity["contract_sha256"],
            )
            stale = make_runner(base / "stale_model", components, spec=spec)
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "initial model state differs",
            ):
                stale.startup(runner_module.InitializationRequest.fresh())

            rng_components = make_components(42)
            rng_spec = make_spec(rng_components)
            rng_runner = make_runner(
                base / "stale_rng",
                rng_components,
                spec=rng_spec,
            )
            random.random()
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "initial global RNG state differs",
            ):
                rng_runner.startup(
                    runner_module.InitializationRequest.fresh()
                )

    def test_optimizer_contract_binds_ordered_parameter_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            components = make_components(42)
            model = components["model"]
            parameters = dict(model.named_parameters())
            first = torch.optim.Adam(
                [
                    {
                        "params": [
                            parameters["0.weight"],
                            parameters["0.bias"],
                        ]
                    },
                    {
                        "params": [
                            parameters["2.weight"],
                            parameters["2.bias"],
                        ]
                    },
                ],
                lr=0.004,
                betas=(0.8, 0.95),
            )
            second = torch.optim.Adam(
                [
                    {
                        "params": [
                            parameters["2.weight"],
                            parameters["2.bias"],
                        ]
                    },
                    {
                        "params": [
                            parameters["0.weight"],
                            parameters["0.bias"],
                        ]
                    },
                ],
                lr=0.004,
                betas=(0.8, 0.95),
            )
            first_contract = runner_module.optimizer_contract(model, first)
            second_contract = runner_module.optimizer_contract(model, second)
            self.assertNotEqual(first_contract, second_contract)
            self.assertEqual(
                first_contract["param_groups"][0]["parameter_names"],
                ["0.weight", "0.bias"],
            )
            self.assertEqual(
                second_contract["param_groups"][0]["parameter_names"],
                ["2.weight", "2.bias"],
            )

            components["optimizer"] = second
            spec = make_spec(
                components,
                optimizer_declaration=first_contract,
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "optimizer contract differs",
            ):
                make_runner(Path(directory) / "run", components, spec=spec)

    def test_selection_policy_is_identity_bound_and_must_match_runner(self) -> None:
        components = make_components(42)
        default = runner_module.pd_miou_selection_policy()
        altered = runner_module.SelectionPolicy(
            primary=runner_module.SelectionRule(
                role=default.primary.role,
                order=(
                    runner_module.MetricOrder("pd", False),
                    *default.primary.order[1:],
                ),
                stored_metrics=default.primary.stored_metrics,
                new_best_field=default.primary.new_best_field,
            ),
            secondary=default.secondary,
        )
        default_spec = make_spec(components)
        altered_spec = make_spec(
            components,
            selection_policy=altered.normalized(),
        )
        default_identity = runner_module.build_run_identity(
            components["model"],
            default_spec,
        )
        altered_identity = runner_module.build_run_identity(
            components["model"],
            altered_spec,
        )
        self.assertNotEqual(
            default_identity["contract_sha256"],
            altered_identity["contract_sha256"],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "selection policy differs",
            ):
                make_runner(
                    Path(directory) / "mismatch",
                    components,
                    spec=default_spec,
                    selection_policy=altered,
                )
            make_runner(
                Path(directory) / "matched",
                components,
                spec=altered_spec,
                selection_policy=altered,
            )

    def test_continuous_four_equals_two_reconstruct_two_exactly(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            continuous_components = make_components(42)
            continuous = make_runner(
                base / "continuous",
                continuous_components,
            )
            continuous.startup(runner_module.InitializationRequest.fresh())
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
            continuous_rng = exact.capture_rng_state(
                continuous_components["loader_generator"]
            )
            continuous_metrics = (
                base / "continuous" / runner_module.METRICS_FILENAME
            ).read_bytes()

            split_components = make_components(42)
            split_spec = make_spec(split_components)
            split_runner = make_runner(
                base / "split",
                split_components,
                spec=split_spec,
            )
            split_runner.startup(runner_module.InitializationRequest.fresh())
            split_trace = [
                run_one_epoch(split_runner, split_components)
                for _ in range(2)
            ]

            rebuilt_components = make_components(999)
            rebuilt_spec = make_spec(
                rebuilt_components,
                initial_model_state_sha256=(
                    split_spec.initial_model_state_sha256
                ),
                initial_rng=split_spec.initial_rng,
                selection_policy=split_spec.selection_policy,
            )
            rebuilt_runner = make_runner(
                base / "split",
                rebuilt_components,
                spec=rebuilt_spec,
            )
            restored = rebuilt_runner.startup(
                runner_module.InitializationRequest.exact()
            )
            self.assertEqual(restored.completed_epoch, 2)
            self.assertEqual(restored.next_epoch, 3)
            split_trace.extend(
                run_one_epoch(rebuilt_runner, rebuilt_components)
                for _ in range(2)
            )
            rebuilt_rng = exact.capture_rng_state(
                rebuilt_components["loader_generator"]
            )

            self.assertEqual(continuous_trace, split_trace)
            self.assertEqual(
                continuous_metrics,
                (base / "split" / runner_module.METRICS_FILENAME).read_bytes(),
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
            self.assertTrue(nested_equal(continuous_rng, rebuilt_rng))
            self.assertEqual(
                continuous_components["scaler"].state_dict(),
                rebuilt_components["scaler"].state_dict(),
            )
            active = rebuilt_runner.snapshot.active
            payload = torch.load(
                active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertIsNone(payload["scheduler"])
            self.assertEqual(payload["epoch"], 4)

    def test_real_dataloader_generator_resumes_at_the_exact_next_order(
        self,
    ) -> None:
        dataset = TensorDataset(torch.arange(12))

        def shuffled_order(components: dict) -> list[int]:
            loader = DataLoader(
                dataset,
                batch_size=3,
                shuffle=True,
                num_workers=0,
                generator=components["loader_generator"],
            )
            return [
                int(value)
                for (batch,) in loader
                for value in batch.tolist()
            ]

        def commit_trace(
            runner: runner_module.ExactRunner,
            order: list[int],
        ) -> None:
            control = runner.next_epoch_control()
            runner.commit_epoch(
                {
                    "train_loss": 1.0 / control.epoch,
                    "loader_trace": order,
                    "pd": 0.70 + 0.01 * control.epoch,
                    "fa": 1.0e-5 / control.epoch,
                    "tiny_pd": 0.60 + 0.01 * control.epoch,
                    "miou": 0.50 + 0.01 * control.epoch,
                    "val_loss": 1.0 / control.epoch,
                }
            )

        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            continuous_components = make_components(42)
            continuous = make_runner(
                base / "continuous",
                continuous_components,
            )
            continuous.startup(runner_module.InitializationRequest.fresh())
            continuous_first = shuffled_order(continuous_components)
            commit_trace(continuous, continuous_first)
            continuous_second = shuffled_order(continuous_components)
            commit_trace(continuous, continuous_second)

            split_components = make_components(42)
            split_spec = make_spec(split_components)
            split = make_runner(
                base / "split",
                split_components,
                spec=split_spec,
            )
            split.startup(runner_module.InitializationRequest.fresh())
            split_first = shuffled_order(split_components)
            commit_trace(split, split_first)
            self.assertEqual(split_first, continuous_first)

            rebuilt_components = make_components(999)
            rebuilt = make_runner(
                base / "split",
                rebuilt_components,
                spec=make_spec(
                    rebuilt_components,
                    initial_model_state_sha256=(
                        split_spec.initial_model_state_sha256
                    ),
                    initial_rng=split_spec.initial_rng,
                    selection_policy=split_spec.selection_policy,
                ),
            )
            restored = rebuilt.startup(
                runner_module.InitializationRequest.exact()
            )
            self.assertEqual(restored.completed_epoch, 1)
            split_second = shuffled_order(rebuilt_components)
            commit_trace(rebuilt, split_second)
            self.assertEqual(split_second, continuous_second)
            self.assertEqual(
                (base / "split" / runner_module.METRICS_FILENAME).read_bytes(),
                (
                    base / "continuous" / runner_module.METRICS_FILENAME
                ).read_bytes(),
            )

    def test_identity_change_is_rejected_on_exact_startup(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            components = make_components(42)
            writer = make_runner(root, components)
            writer.startup(runner_module.InitializationRequest.fresh())
            run_one_epoch(writer, components)

            rebuilt = make_components(123)
            changed_spec = make_spec(rebuilt, batch_size=3)
            reader = make_runner(root, rebuilt, spec=changed_spec)
            with self.assertRaisesRegex(Exception, "run identity mismatch"):
                reader.startup(runner_module.InitializationRequest.exact())

    def test_optimizer_and_scaler_declarations_match_real_initial_objects(
        self,
    ) -> None:
        base = make_components(42)
        base_optimizer = runner_module.optimizer_contract(
            base["model"],
            base["optimizer"],
        )
        base_scaler = runner_module.scaler_contract(
            base["scaler"],
            amp=False,
        )

        def remove_last_group_parameter(value: dict) -> None:
            group = value["param_groups"][0]
            group["parameter_count"] -= 1
            group["parameter_names"].pop()

        optimizer_mutations = {
            "betas": lambda value: value["defaults"].__setitem__(
                "betas",
                [0.7, 0.95],
            ),
            "eps": lambda value: value["defaults"].__setitem__("eps", 1e-6),
            "weight_decay": lambda value: value["param_groups"][0][
                "options"
            ].__setitem__("weight_decay", 0.01),
            "param_group": remove_last_group_parameter,
        }
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory)
            for name, mutate in optimizer_mutations.items():
                with self.subTest(optimizer=name):
                    components = make_components(42)
                    declaration = copy.deepcopy(base_optimizer)
                    mutate(declaration)
                    spec = make_spec(
                        components,
                        optimizer_declaration=declaration,
                    )
                    with self.assertRaisesRegex(
                        runner_module.ExactRunnerError,
                        "optimizer contract differs",
                    ):
                        make_runner(
                            base_path / f"optimizer_{name}",
                            components,
                            spec=spec,
                        )

            scaler_mutations = {
                "class": lambda value: value.__setitem__(
                    "class",
                    "fixture.DifferentScaler",
                ),
                "state": lambda value: value["initial_state"].__setitem__(
                    "scale",
                    256.0,
                ),
                "config": lambda value: value["config"].__setitem__(
                    "fixture",
                    "different",
                ),
            }
            for name, mutate in scaler_mutations.items():
                with self.subTest(scaler=name):
                    components = make_components(42)
                    declaration = copy.deepcopy(base_scaler)
                    mutate(declaration)
                    spec = make_spec(
                        components,
                        scaler_declaration=declaration,
                    )
                    with self.assertRaisesRegex(
                        runner_module.ExactRunnerError,
                        "scaler contract differs",
                    ):
                        make_runner(
                            base_path / f"scaler_{name}",
                            components,
                            spec=spec,
                        )

    def test_prestepped_optimizer_wrong_loader_seed_and_nonbase_lr_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            stepped = make_components(42)
            stepped_spec = make_spec(stepped)
            stepped["optimizer"].zero_grad(set_to_none=True)
            stepped["model"](torch.ones(1, 3)).sum().backward()
            stepped["optimizer"].step()
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "empty step state",
            ):
                make_runner(base / "stepped", stepped, spec=stepped_spec)

            wrong_loader = make_components(42)
            wrong_loader["loader_generator"].manual_seed(43)
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "manual_seed\\(spec.seed\\)",
            ):
                make_runner(base / "loader", wrong_loader)

            wrong_lr = make_components(42)
            wrong_lr_spec = make_spec(wrong_lr)
            wrong_lr["optimizer"].param_groups[0]["lr"] = 0.003
            wrong_lr_declaration = runner_module.optimizer_contract(
                wrong_lr["model"],
                wrong_lr["optimizer"]
            )
            wrong_lr_spec = make_spec(
                wrong_lr,
                optimizer_declaration=wrong_lr_declaration,
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "must equal manual schedule base_lr",
            ):
                make_runner(base / "lr", wrong_lr, spec=wrong_lr_spec)

    def test_commit_rejects_optimizer_lr_drift_from_open_control(self) -> None:
        fields = {
            "train_loss": 0.5,
            "pd": 0.8,
            "fa": 1e-6,
            "tiny_pd": 0.7,
            "miou": 0.6,
            "val_loss": 0.5,
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            components = make_components(42)
            runner = make_runner(Path(directory) / "run", components)
            runner.startup(runner_module.InitializationRequest.fresh())
            control = runner.next_epoch_control()
            components["optimizer"].param_groups[0]["lr"] = (
                control.learning_rate / 2.0
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "LR differs from the open epoch control",
            ):
                runner.commit_epoch(fields)
            self.assertEqual(runner.snapshot.completed_epoch, 0)
            components["optimizer"].param_groups[0]["lr"] = (
                control.learning_rate
            )
            committed = runner.commit_epoch(fields)
            self.assertEqual(committed.completed_epoch, 1)

    def test_initialization_mode_and_parent_digest_are_identity_bound(self) -> None:
        provenance = {
            "schema": "sctransnet_tpd_extension_warm_start_v1",
            "parent_checkpoint_path": "/fixture/parent.pth.tar",
            "parent_checkpoint_sha256": "a" * 64,
            "parent_state_dict_path": ["state_dict"],
            "parent_state_key_count": 4,
            "preserved_new_state_key_count": 2,
            "new_module_prefixes": ["new"],
            "zero_init_prefixes": [],
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = make_components(42)
            parent_identity = runner_module.build_run_identity(
                parent["model"],
                make_spec(parent, run_id="parent"),
            )
            payload_a = exact.build_parent_warm_start_checkpoint(
                parent_model=parent["model"],
                parent_epoch=3,
                parent_identity=parent_identity,
                extra_state={"source": "a"},
            )
            payload_b = exact.build_parent_warm_start_checkpoint(
                parent_model=parent["model"],
                parent_epoch=3,
                parent_identity=parent_identity,
                extra_state={"source": "b"},
            )
            path_a = root / "parent_a.pth"
            path_b = root / "parent_b.pth"
            torch.save(payload_a, path_a)
            torch.save(payload_b, path_b)
            sha_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
            sha_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
            self.assertNotEqual(sha_a, sha_b)
            child = make_components(42)
            prepared_a = runner_module.prepare_same_layout_parent(
                path_a,
                child_model=child["model"],
                expected_parent_checkpoint_sha256=sha_a,
                expected_parent_identity=parent_identity,
                expected_parent_epoch=3,
            )
            prepared_b = runner_module.prepare_same_layout_parent(
                path_b,
                child_model=child["model"],
                expected_parent_checkpoint_sha256=sha_b,
                expected_parent_identity=parent_identity,
                expected_parent_epoch=3,
            )
            loaded_child_sha256 = (
                prepared_a.loaded_child_model_state_sha256
            )
            child_spec = make_spec(
                child,
                initialization_contract=(
                    prepared_a.initialization_contract()
                ),
                initial_model_state_sha256=loaded_child_sha256,
            )
            digest_runner = make_runner(
                root / "digest_child",
                child,
                spec=child_spec,
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "initialization request differs",
            ):
                digest_runner.startup(
                    runner_module.InitializationRequest.parent(prepared_b)
                )

            fresh_child = make_components(42)
            mode_runner = make_runner(root / "mode_child", fresh_child)
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "initialization request differs",
            ):
                mode_runner.startup(
                    runner_module.InitializationRequest.extension_parent(
                        provenance,
                        loaded_child_model_state_sha256=(
                            runner_module.initial_model_state_sha256(
                                fresh_child["model"]
                            )
                        ),
                    )
                )

            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "checkpoint SHA-256 mismatch",
            ):
                runner_module.prepare_same_layout_parent(
                    path_a,
                    child_model=make_components(42)["model"],
                    expected_parent_checkpoint_sha256="f" * 64,
                    expected_parent_identity=parent_identity,
                    expected_parent_epoch=3,
                )

    def test_adapter_and_active_journal_repair_all_derived_views(self) -> None:
        calls: list[tuple[str, int]] = []

        def adapter(
            context: runner_module.CompatibilityPayloadContext,
        ) -> dict:
            calls.append((context.role, context.epoch))
            exact_payload = context.exact_payload
            return {
                "derived_schema": runner_module.DERIVED_CHECKPOINT_SCHEMA,
                "checkpoint_role": context.role,
                "epoch": context.epoch,
                "variant": context.run_identity["variant"],
                "dataset": context.run_identity["dataset"],
                "seed": context.run_identity["seed"],
                "state_dict": copy.deepcopy(
                    exact_payload["model"]["state_dict"]
                ),
                "optimizer": copy.deepcopy(
                    exact_payload["optimizer"]["state_dict"]
                ),
                "scaler": copy.deepcopy(
                    exact_payload["scaler"]["state_dict"]
                ),
                "validation_metrics": context.metrics,
                "run_identity": context.run_identity,
                "adapter_marker": context.event["epoch"],
            }

        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            components = make_components(42)
            runner = make_runner(root, components, adapter=adapter)
            runner.startup(runner_module.InitializationRequest.fresh())
            run_one_epoch(runner, components)
            run_one_epoch(runner, components)
            active_metrics = runner.snapshot.active.metrics_path.read_bytes()

            for name in (
                runner_module.METRICS_FILENAME,
                runner_module.LAST_FILENAME,
                runner_module.BEST_FILENAME,
                runner_module.BEST_MIOU_FILENAME,
            ):
                (root / name).write_bytes(b"corrupt")
            runner.repair_derived_artifacts()

            self.assertEqual(
                (root / runner_module.METRICS_FILENAME).read_bytes(),
                active_metrics,
            )
            expected = {
                runner_module.LAST_FILENAME: "last_completed_epoch",
                runner_module.BEST_FILENAME: "best_validation_pd_primary",
                runner_module.BEST_MIOU_FILENAME:
                    "best_validation_miou_secondary",
            }
            for name, role in expected.items():
                payload = torch.load(
                    root / name,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(payload["checkpoint_role"], role)
                self.assertEqual(payload["epoch"], 2)
                self.assertEqual(payload["adapter_marker"], 2)
                self.assertIn("state_dict", payload)
            self.assertIn(("last_completed_epoch", 2), calls)

    def test_adapter_cannot_change_exact_model_optimizer_or_scaler_state(
        self,
    ) -> None:
        def adapter_for(changed_field: str):
            def adapter(
                context: runner_module.CompatibilityPayloadContext,
            ) -> dict:
                payload = {
                    "checkpoint_role": context.role,
                    "epoch": context.epoch,
                    "state_dict": copy.deepcopy(
                        context.exact_payload["model"]["state_dict"]
                    ),
                    "optimizer": copy.deepcopy(
                        context.exact_payload["optimizer"]["state_dict"]
                    ),
                    "scaler": copy.deepcopy(
                        context.exact_payload["scaler"]["state_dict"]
                    ),
                    "validation_metrics": context.metrics,
                    "run_identity": context.run_identity,
                }
                if changed_field == "state_dict":
                    tensor = next(iter(payload["state_dict"].values()))
                    tensor.add_(1.0)
                elif changed_field == "optimizer":
                    payload["optimizer"]["param_groups"][0]["lr"] += 1.0
                else:
                    payload["scaler"]["updates"] += 1
                return payload

            return adapter

        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for changed_field in ("state_dict", "optimizer", "scaler"):
                with self.subTest(changed_field=changed_field):
                    components = make_components(42)
                    runner = make_runner(
                        base / changed_field,
                        components,
                        adapter=adapter_for(changed_field),
                    )
                    runner.startup(
                        runner_module.InitializationRequest.fresh()
                    )
                    with self.assertRaisesRegex(
                        runner_module.ExactRunnerError,
                        f"changed exact source {changed_field}",
                    ):
                        run_one_epoch(runner, components)
                    self.assertEqual(runner.snapshot.completed_epoch, 1)
                    self.assertTrue(
                        runner.snapshot.derived_artifacts_dirty
                    )
                    with self.assertRaisesRegex(
                        runner_module.ExactRunnerError,
                        "must be repaired",
                    ):
                        runner.next_epoch_control()

    def test_adapter_schema_is_injected_and_old_best_remains_valid(self) -> None:
        calls: list[tuple[str, int]] = []

        def adapter(
            context: runner_module.CompatibilityPayloadContext,
        ) -> dict:
            calls.append((context.role, context.epoch))
            return {
                # The runner, not every thin entry, owns this schema marker.
                "checkpoint_role": context.role,
                "epoch": context.epoch,
                "state_dict": copy.deepcopy(
                    context.exact_payload["model"]["state_dict"]
                ),
                "optimizer": copy.deepcopy(
                    context.exact_payload["optimizer"]["state_dict"]
                ),
                "scaler": copy.deepcopy(
                    context.exact_payload["scaler"]["state_dict"]
                ),
                "run_identity": context.run_identity,
                "validation_metrics": context.metrics,
            }

        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            components = make_components(42)
            runner = make_runner(root, components, adapter=adapter)
            runner.startup(runner_module.InitializationRequest.fresh())
            for epoch in range(1, 5):
                runner.next_epoch_control()
                runner.commit_epoch(
                    {
                        "train_loss": float(epoch),
                        "pd": 0.90 - 0.01 * epoch,
                        "fa": 1.0e-6 + epoch * 1.0e-7,
                        "tiny_pd": 0.80 - 0.01 * epoch,
                        "miou": 0.85 - 0.01 * epoch,
                        "val_loss": 0.10 + 0.01 * epoch,
                    }
                )
            best_path = root / runner_module.BEST_FILENAME
            best = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(
                best["derived_schema"],
                runner_module.DERIVED_CHECKPOINT_SCHEMA,
            )
            for field in (
                "source_exact_checkpoint_sha256",
                "state_dict_sha256",
                "optimizer_state_sha256",
                "scaler_state_sha256",
            ):
                self.assertEqual(len(best[field]), 64)
            best_calls_before = [
                call for call in calls if call[0] == "best_validation_pd_primary"
            ]
            self.assertEqual(best_calls_before, [("best_validation_pd_primary", 1)])

            # Epoch 1 is outside the two journal slots by now. A valid derived
            # best must be recognized and retained, rather than spuriously
            # requesting an unavailable source epoch.
            runner.repair_derived_artifacts()
            best_calls_after = [
                call for call in calls if call[0] == "best_validation_pd_primary"
            ]
            self.assertEqual(best_calls_after, best_calls_before)
            self.assertEqual(
                torch.load(
                    best_path,
                    map_location="cpu",
                    weights_only=False,
                )["epoch"],
                1,
            )

            corrupted = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
            next(iter(corrupted["state_dict"].values())).add_(1.0)
            torch.save(corrupted, best_path)
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "no longer retains epoch 1",
            ):
                runner.repair_derived_artifacts()
            self.assertTrue(runner.snapshot.derived_artifacts_dirty)

    def test_parent_loads_weights_then_starts_a_fresh_journal(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            source = make_components(42)
            source_spec = make_spec(source, run_id="parent-run")
            parent_identity = runner_module.build_run_identity(
                source["model"],
                source_spec,
            )
            with torch.no_grad():
                for parameter in source["model"].parameters():
                    parameter.add_(0.25)
            expected_weights = copy.deepcopy(source["model"].state_dict())
            parent_payload = exact.build_parent_warm_start_checkpoint(
                parent_model=source["model"],
                parent_epoch=7,
                parent_identity=parent_identity,
                extra_state={"source": "parent"},
            )
            parent_checkpoint = Path(directory) / "parent.exact.pth"
            torch.save(parent_payload, parent_checkpoint)
            parent_sha256 = hashlib.sha256(
                parent_checkpoint.read_bytes()
            ).hexdigest()
            child = make_components(999)
            prepared_parent = runner_module.prepare_same_layout_parent(
                parent_checkpoint,
                child_model=child["model"],
                expected_parent_checkpoint_sha256=parent_sha256,
                expected_parent_identity=parent_identity,
                expected_parent_epoch=7,
            )
            loaded_child_sha256 = (
                prepared_parent.loaded_child_model_state_sha256
            )
            child_spec = make_spec(
                child,
                run_id="child-run",
                initialization_contract=(
                    prepared_parent.initialization_contract()
                ),
                initial_model_state_sha256=loaded_child_sha256,
            )
            rng_before = exact.capture_rng_state(child["loader_generator"])
            runner = make_runner(
                Path(directory) / "child",
                child,
                spec=child_spec,
            )
            started = runner.startup(
                runner_module.InitializationRequest.parent(prepared_parent)
            )
            rng_after = exact.capture_rng_state(child["loader_generator"])
            self.assertEqual(
                started.initialization_mode,
                exact.InitializationMode.PARENT_WARM_START,
            )
            self.assertEqual(started.completed_epoch, 0)
            self.assertIsNone(started.active)
            self.assertEqual(started.parent_provenance["parent_epoch"], 7)
            self.assertTrue(
                nested_equal(expected_weights, child["model"].state_dict())
            )
            self.assertEqual(child["optimizer"].state_dict()["state"], {})
            self.assertTrue(nested_equal(rng_before, rng_after))

            run_one_epoch(runner, child)
            active_payload = torch.load(
                runner.snapshot.active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(active_payload["mode"], exact.EXACT_RESUME_MODE)
            provenance = active_payload["extra_state"][
                "initialization_provenance"
            ]
            self.assertEqual(
                provenance["initial_mode"],
                exact.PARENT_WARM_START_MODE,
            )
            self.assertEqual(provenance["parent"]["parent_epoch"], 7)

    def test_same_layout_parent_is_verified_and_loaded_from_one_read(
        self,
    ) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_a = make_components(42)
            parent_identity = runner_module.build_run_identity(
                source_a["model"],
                make_spec(source_a, run_id="parent"),
            )
            state_a = copy.deepcopy(source_a["model"].state_dict())
            payload_a = exact.build_parent_warm_start_checkpoint(
                parent_model=source_a["model"],
                parent_epoch=5,
                parent_identity=parent_identity,
            )
            source_b = make_components(999)
            payload_b = exact.build_parent_warm_start_checkpoint(
                parent_model=source_b["model"],
                parent_epoch=5,
                parent_identity=parent_identity,
            )
            parent_path = base / "parent.pth"
            replacement_path = base / "replacement.pth"
            torch.save(payload_a, parent_path)
            torch.save(payload_b, replacement_path)
            digest_a = hashlib.sha256(parent_path.read_bytes()).hexdigest()
            replacement_bytes = replacement_path.read_bytes()
            child = make_components(123)
            real_read = runner_module._read_regular
            parent_reads = 0

            def swap_after_read(path: Path, label: str) -> bytes:
                nonlocal parent_reads
                content = real_read(path, label)
                if Path(path) == parent_path:
                    parent_reads += 1
                    parent_path.write_bytes(replacement_bytes)
                return content

            with mock.patch.object(
                runner_module,
                "_read_regular",
                side_effect=swap_after_read,
            ):
                prepared_parent = runner_module.prepare_same_layout_parent(
                    parent_path,
                    child_model=child["model"],
                    expected_parent_checkpoint_sha256=digest_a,
                    expected_parent_identity=parent_identity,
                    expected_parent_epoch=5,
                )
                loaded_child_sha256 = (
                    prepared_parent.loaded_child_model_state_sha256
                )
                child_spec = make_spec(
                    child,
                    initialization_contract=(
                        prepared_parent.initialization_contract()
                    ),
                    initial_model_state_sha256=loaded_child_sha256,
                )
                runner = make_runner(
                    base / "child",
                    child,
                    spec=child_spec,
                )
                runner.startup(
                    runner_module.InitializationRequest.parent(
                        prepared_parent
                    )
                )
            self.assertEqual(parent_reads, 1)
            self.assertTrue(nested_equal(child["model"].state_dict(), state_a))
            self.assertNotEqual(
                hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                digest_a,
            )

    def test_external_extension_parent_is_fresh_with_strict_provenance(self) -> None:
        provenance = {
            "schema": "sctransnet_tpd_extension_warm_start_v1",
            "parent_checkpoint_path": "/fixture/parent.pth.tar",
            "parent_checkpoint_sha256": "a" * 64,
            "parent_state_dict_path": ["state_dict"],
            "parent_state_key_count": 4,
            "preserved_new_state_key_count": 2,
            "new_module_prefixes": ["tpd_frequency_gate"],
            "zero_init_prefixes": ["tpd_frequency_gate.alpha"],
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            components = make_components(42)
            # Represents the already completed strict extension transfer.
            with torch.no_grad():
                for parameter in components["model"].parameters():
                    parameter.add_(0.125)
            state_before = copy.deepcopy(components["model"].state_dict())
            loaded_child_sha256 = (
                runner_module.initial_model_state_sha256(
                    components["model"]
                )
            )
            rng_before = exact.capture_rng_state(
                components["loader_generator"]
            )
            child_spec = make_spec(
                components,
                initialization_contract=(
                    runner_module.extension_parent_initialization_contract(
                        provenance,
                        loaded_child_model_state_sha256=loaded_child_sha256,
                    )
                ),
            )
            runner = make_runner(
                Path(directory) / "run",
                components,
                spec=child_spec,
            )
            started = runner.startup(
                runner_module.InitializationRequest.extension_parent(
                    provenance,
                    loaded_child_model_state_sha256=loaded_child_sha256,
                )
            )
            self.assertEqual(
                started.initialization_mode,
                exact.InitializationMode.PARENT_WARM_START,
            )
            self.assertEqual(started.completed_epoch, 0)
            self.assertIsNone(started.active)
            self.assertEqual(
                started.parent_provenance["mode"],
                runner_module.EXTENSION_PARENT_MODE,
            )
            self.assertEqual(
                started.parent_provenance["extension_warm_start"],
                provenance,
            )
            self.assertTrue(
                nested_equal(state_before, components["model"].state_dict())
            )
            self.assertTrue(
                nested_equal(
                    rng_before,
                    exact.capture_rng_state(components["loader_generator"]),
                )
            )
            self.assertEqual(components["optimizer"].state_dict()["state"], {})

            run_one_epoch(runner, components)
            active_payload = torch.load(
                runner.snapshot.active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            lineage = active_payload["extra_state"][
                "initialization_provenance"
            ]
            self.assertEqual(
                lineage["initial_mode"],
                runner_module.EXTENSION_PARENT_MODE,
            )
            self.assertEqual(
                lineage["parent"]["extension_warm_start"],
                provenance,
            )

    def test_extension_parent_rejects_changed_loaded_child_state(self) -> None:
        provenance = {
            "schema": "sctransnet_tpd_extension_warm_start_v1",
            "parent_checkpoint_path": "/fixture/parent.pth.tar",
            "parent_checkpoint_sha256": "a" * 64,
            "parent_state_dict_path": ["state_dict"],
            "parent_state_key_count": 4,
            "preserved_new_state_key_count": 2,
            "new_module_prefixes": ["tpd_frequency_gate"],
            "zero_init_prefixes": ["tpd_frequency_gate.alpha"],
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            components = make_components(42)
            loaded_child_sha256 = (
                runner_module.initial_model_state_sha256(
                    components["model"]
                )
            )
            spec = make_spec(
                components,
                initialization_contract=(
                    runner_module.extension_parent_initialization_contract(
                        provenance,
                        loaded_child_model_state_sha256=loaded_child_sha256,
                    )
                ),
                initial_model_state_sha256=loaded_child_sha256,
            )
            with torch.no_grad():
                next(components["model"].parameters()).add_(0.5)
            runner = make_runner(
                Path(directory) / "run",
                components,
                spec=spec,
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "initial model state differs",
            ):
                runner.startup(
                    runner_module.InitializationRequest.extension_parent(
                        provenance,
                        loaded_child_model_state_sha256=loaded_child_sha256,
                    )
                )

    def test_empty_journal_rejects_old_trajectory_views_for_all_fresh_modes(
        self,
    ) -> None:
        extension_provenance = {
            "schema": "sctransnet_tpd_extension_warm_start_v1",
            "parent_checkpoint_path": "/fixture/parent.pth.tar",
            "parent_checkpoint_sha256": "a" * 64,
            "parent_state_dict_path": ["state_dict"],
            "parent_state_key_count": 4,
            "preserved_new_state_key_count": 2,
            "new_module_prefixes": ["new"],
            "zero_init_prefixes": [],
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            parent_components = make_components(42)
            parent_spec = make_spec(parent_components, run_id="parent")
            parent_identity = runner_module.build_run_identity(
                parent_components["model"],
                parent_spec,
            )
            parent_payload = exact.build_parent_warm_start_checkpoint(
                parent_model=parent_components["model"],
                parent_epoch=3,
                parent_identity=parent_identity,
            )
            parent_checkpoint = base / "parent.exact.pth"
            torch.save(parent_payload, parent_checkpoint)
            parent_sha256 = hashlib.sha256(
                parent_checkpoint.read_bytes()
            ).hexdigest()
            prepared_parent = runner_module.prepare_same_layout_parent(
                parent_checkpoint,
                child_model=make_components(42)["model"],
                expected_parent_checkpoint_sha256=parent_sha256,
                expected_parent_identity=parent_identity,
                expected_parent_epoch=3,
            )
            loaded_child_sha256 = (
                prepared_parent.loaded_child_model_state_sha256
            )
            cases = (
                (
                    "fresh",
                    runner_module.InitializationRequest.fresh(),
                    runner_module.fresh_initialization_contract(),
                ),
                (
                    "same_parent",
                    runner_module.InitializationRequest.parent(prepared_parent),
                    prepared_parent.initialization_contract(),
                ),
                (
                    "extension_parent",
                    runner_module.InitializationRequest.extension_parent(
                        extension_provenance,
                        loaded_child_model_state_sha256=loaded_child_sha256,
                    ),
                    runner_module.extension_parent_initialization_contract(
                        extension_provenance,
                        loaded_child_model_state_sha256=loaded_child_sha256,
                    ),
                ),
            )
            for name, request, initialization_contract in cases:
                with self.subTest(name=name):
                    components = make_components(42)
                    root = base / name
                    root.mkdir()
                    (root / "protocol.json").write_text("{}\n")
                    (root / "split.json").write_text("{}\n")
                    (root / runner_module.METRICS_FILENAME).write_text(
                        '{"epoch":1}\n'
                    )
                    runner = make_runner(
                        root,
                        components,
                        spec=make_spec(
                            components,
                            initialization_contract=initialization_contract,
                            initial_model_state_sha256=loaded_child_sha256,
                        ),
                    )
                    with self.assertRaisesRegex(
                        runner_module.ExactRunnerError,
                        "existing derived trajectory artifacts",
                    ):
                        runner.startup(request)

            allowed_components = make_components(42)
            allowed_root = base / "protocol_split_only"
            allowed_root.mkdir()
            (allowed_root / "protocol.json").write_text("{}\n")
            (allowed_root / "split.json").write_text("{}\n")
            allowed = make_runner(allowed_root, allowed_components)
            snapshot = allowed.startup(
                runner_module.InitializationRequest.fresh()
            )
            self.assertEqual(snapshot.completed_epoch, 0)

    def test_marker_failure_retry_and_random_drift_guard(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "retry"
            components = make_components(42)
            runner = make_runner(root, components)
            runner.startup(runner_module.InitializationRequest.fresh())
            runner.next_epoch_control()
            real_atomic = journal_module._atomic_write_bytes

            def fail_marker(destination: Path, content: bytes) -> None:
                if destination.name == journal_module.MARKER_FILENAME:
                    raise OSError("injected marker failure")
                real_atomic(destination, content)

            fields = {
                "train_loss": 0.5,
                "pd": 0.8,
                "fa": 1e-6,
                "tiny_pd": 0.7,
                "miou": 0.6,
                "val_loss": 0.5,
            }
            with mock.patch.object(
                journal_module,
                "_atomic_write_bytes",
                side_effect=fail_marker,
            ):
                with self.assertRaisesRegex(OSError, "marker failure"):
                    runner.commit_epoch(fields)
            self.assertEqual(runner.snapshot.completed_epoch, 0)
            committed = runner.retry_pending_commit()
            self.assertEqual(committed.completed_epoch, 1)

            drift_root = Path(directory) / "drift"
            drift_components = make_components(42)
            drift = make_runner(drift_root, drift_components)
            drift.startup(runner_module.InitializationRequest.fresh())
            drift.next_epoch_control()
            with mock.patch.object(
                journal_module,
                "_atomic_write_bytes",
                side_effect=fail_marker,
            ):
                with self.assertRaisesRegex(OSError, "marker failure"):
                    drift.commit_epoch(fields)
            random.random()
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "random streams changed",
            ):
                drift.retry_pending_commit()

    def test_retry_adopts_epoch_when_marker_switched_before_write_raised(
        self,
    ) -> None:
        fields = {
            "train_loss": 0.5,
            "pd": 0.8,
            "fa": 1e-6,
            "tiny_pd": 0.7,
            "miou": 0.6,
            "val_loss": 0.5,
        }
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            components = make_components(42)
            runner = make_runner(root, components)
            runner.startup(runner_module.InitializationRequest.fresh())
            runner.next_epoch_control()
            real_atomic = journal_module._atomic_write_bytes

            def write_marker_then_raise(
                destination: Path,
                content: bytes,
            ) -> None:
                real_atomic(destination, content)
                if destination.name == journal_module.MARKER_FILENAME:
                    raise OSError("marker switched before injected failure")

            with mock.patch.object(
                journal_module,
                "_atomic_write_bytes",
                side_effect=write_marker_then_raise,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "marker switched",
                ):
                    runner.commit_epoch(fields)
            self.assertEqual(runner.snapshot.completed_epoch, 0)
            self.assertEqual(runner.journal.load_active().epoch, 1)

            adopted = runner.retry_pending_commit()
            self.assertEqual(adopted.completed_epoch, 1)
            self.assertFalse(adopted.derived_artifacts_dirty)
            self.assertTrue((root / runner_module.LAST_FILENAME).is_file())
            self.assertEqual(
                len(
                    (
                        root / runner_module.METRICS_FILENAME
                    ).read_text().splitlines()
                ),
                1,
            )

    def test_post_commit_publication_failure_is_repairable(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            components = make_components(42)
            runner = make_runner(root, components)
            runner.startup(runner_module.InitializationRequest.fresh())
            real_save = exact.atomic_torch_save

            def fail_last(payload, destination, *args, **kwargs):
                if Path(destination).name == runner_module.LAST_FILENAME:
                    raise OSError("injected derived failure")
                return real_save(payload, destination, *args, **kwargs)

            with mock.patch.object(
                exact,
                "atomic_torch_save",
                side_effect=fail_last,
            ):
                with self.assertRaisesRegex(
                    runner_module.ExactRunnerError,
                    "epoch is committed",
                ):
                    run_one_epoch(runner, components)
            self.assertEqual(runner.snapshot.completed_epoch, 1)
            self.assertTrue(runner.snapshot.derived_artifacts_dirty)
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "must be repaired",
            ):
                runner.next_epoch_control()
            runner.repair_derived_artifacts()
            self.assertFalse(runner.snapshot.derived_artifacts_dirty)
            self.assertTrue((root / runner_module.LAST_FILENAME).is_file())
            self.assertTrue((root / runner_module.BEST_FILENAME).is_file())

    def test_exact_startup_repairs_without_allowing_adapter_rng_drift(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            source_components = make_components(42)
            source_spec = make_spec(source_components)
            source = make_runner(
                root,
                source_components,
                spec=source_spec,
            )
            source.startup(runner_module.InitializationRequest.fresh())
            run_one_epoch(source, source_components)
            exact_payload = torch.load(
                source.snapshot.active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_rng = copy.deepcopy(exact_payload["rng_state"])

            rebuilt = make_components(999)

            def consuming_adapter(
                context: runner_module.CompatibilityPayloadContext,
            ) -> dict:
                random.random()
                np.random.rand()
                torch.rand(1)
                torch.rand(1, generator=rebuilt["loader_generator"])
                return {
                    "checkpoint_role": context.role,
                    "epoch": context.epoch,
                    "state_dict": copy.deepcopy(
                        context.exact_payload["model"]["state_dict"]
                    ),
                    "optimizer": copy.deepcopy(
                        context.exact_payload["optimizer"]["state_dict"]
                    ),
                    "scaler": copy.deepcopy(
                        context.exact_payload["scaler"]["state_dict"]
                    ),
                    "validation_metrics": context.metrics,
                    "run_identity": context.run_identity,
                }

            resumed = make_runner(
                root,
                rebuilt,
                spec=make_spec(
                    rebuilt,
                    initial_model_state_sha256=(
                        source_spec.initial_model_state_sha256
                    ),
                    initial_rng=source_spec.initial_rng,
                    selection_policy=source_spec.selection_policy,
                ),
                adapter=consuming_adapter,
            )
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "consumed random streams",
            ):
                resumed.startup(runner_module.InitializationRequest.exact())
            actual_rng = exact.capture_rng_state(
                rebuilt["loader_generator"]
            )
            self.assertTrue(nested_equal(actual_rng, expected_rng))
            self.assertEqual(resumed.snapshot.completed_epoch, 1)

    def test_manual_schedule_workers_and_runner_owned_event_fields(self) -> None:
        self.assertIn("ManualCosineSchedule", runner_module.__all__)
        components = make_components(42)
        with self.assertRaisesRegex(
            runner_module.ExactRunnerError,
            "workers=0",
        ):
            make_spec(components, workers=1).normalized()
        schedule = runner_module.ManualCosineSchedule(
            total_epochs=4,
            base_lr=0.004,
            min_lr=0.0001,
            warmup_epochs=1,
        )
        self.assertEqual(schedule.learning_rate(1), 0.004)
        self.assertAlmostEqual(schedule.learning_rate(4), 0.0001)

        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            runner = make_runner(Path(directory) / "run", components)
            runner.startup(runner_module.InitializationRequest.fresh())
            runner.next_epoch_control()
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "control has not been committed",
            ):
                runner.next_epoch_control()
            with self.assertRaisesRegex(
                runner_module.ExactRunnerError,
                "runner-owned keys",
            ):
                runner.commit_epoch(
                    {
                        "epoch": 1,
                        "pd": 0.8,
                        "fa": 1e-6,
                        "tiny_pd": 0.7,
                        "miou": 0.6,
                        "val_loss": 0.5,
                    }
                )


if __name__ == "__main__":
    unittest.main()
