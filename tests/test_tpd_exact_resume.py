from __future__ import annotations

import copy
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
from torch.utils.data import DataLoader, Dataset

from experiments import tpd_exact_resume as exact


class StateScaler:
    """Small CPU-only scaler fixture with the GradScaler state API."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


class RandomTraceDataset(Dataset):
    def __len__(self) -> int:
        return 9

    def __getitem__(self, index: int) -> tuple[int, int, int, int]:
        return (
            index,
            random.getrandbits(31),
            int(np.random.randint(0, 2**31 - 1)),
            int(torch.randint(0, 2**31 - 1, ()).item()),
        )


@contextmanager
def cpu_rng_environment():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
        mock.patch.object(torch.cuda, "get_rng_state_all") as get_cuda_state,
        mock.patch.object(torch.cuda, "set_rng_state_all") as set_cuda_state,
    ):
        yield get_cuda_state, set_cuda_state


def collect_loader_trace(
    loader: DataLoader,
) -> list[tuple[int, int, int, int]]:
    trace: list[tuple[int, int, int, int]] = []
    for indices, python_values, numpy_values, torch_values in loader:
        trace.extend(
            zip(
                indices.tolist(),
                python_values.tolist(),
                numpy_values.tolist(),
                torch_values.tolist(),
            )
        )
    return trace


def optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]])
    model(inputs).square().mean().backward()
    optimizer.step()


class ExactResumeFixture:
    def __init__(self) -> None:
        torch.manual_seed(77)
        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        optimizer_step(self.model, self.optimizer)
        self.scaler = StateScaler(512.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=0.5,
        )
        optimizer_step(self.model, self.optimizer)
        self.scheduler.step()
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(991)
        self.identity = {
            "run_id": "fixture-run",
            "variant": "tpd_fixture",
            "architecture_id": "tpd_fixture_v1",
            "dataset": "fixture",
            "seed": 42,
            "split_seed": 3407,
            "split_sha256": "a" * 64,
            "source_lock_sha256": "b" * 64,
        }
        metrics = {
            "pd": 0.9,
            "fa": 1.0e-6,
            "miou": 0.8,
            "val_loss": 0.2,
            "tiny_pd": 1.0,
            "tiny_target_count": 3,
        }
        self.selection = {
            "primary": {
                "role": "best_validation_pd_primary",
                "epoch": 3,
                "key": [0.9, -1.0e-6, 1.0, 0.8, -0.2],
                "metrics": metrics,
            },
            "secondary": {
                "role": "best_validation_miou_secondary",
                "epoch": 2,
                "key": [0.8, 0.9, -1.0e-6, 1.0, -0.2],
                "metrics": metrics,
            },
        }
        self.boundary = {
            "completed_epoch": 3,
            "event_count": 3,
            "last_event_epoch": 3,
            "metrics_sha256": "c" * 64,
            "last_event_sha256": "d" * 64,
        }

        random.seed(123)
        np.random.seed(456)
        torch.manual_seed(789)
        self.checkpoint = exact.build_exact_resume_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            epoch=3,
            run_identity=self.identity,
            best_selection=self.selection,
            metrics_boundary=self.boundary,
            loader_generator=self.loader_generator,
            extra_state={"skipped_singleton_batches": 2},
        )

    def restore(self, checkpoint: object | None = None) -> exact.ExactResumeResult:
        return exact.restore_exact_resume(
            self.checkpoint if checkpoint is None else checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.scheduler,
            loader_generator=self.loader_generator,
            expected_run_identity=self.identity,
            expected_epoch=3,
            expected_metrics_boundary=self.boundary,
            expected_best_selection=self.selection,
        )


class TPDExactResumeTests(unittest.TestCase):
    def test_rng_and_dataloader_trajectory_restore_bitwise_on_cpu(self) -> None:
        with cpu_rng_environment() as (get_cuda_state, set_cuda_state):
            fixture = ExactResumeFixture()
            loader = DataLoader(
                RandomTraceDataset(),
                batch_size=4,
                shuffle=True,
                num_workers=0,
                generator=fixture.loader_generator,
            )
            expected_loader = collect_loader_trace(loader)
            expected_tail = (
                random.getrandbits(63),
                np.random.randint(0, 2**63 - 1, dtype=np.int64),
                torch.randint(0, 2**63 - 1, (5,), dtype=torch.int64),
            )

            for parameter in fixture.model.parameters():
                parameter.data.add_(10.0)
            fixture.optimizer.param_groups[0]["lr"] = 0.9
            fixture.scaler.scale = 3.0
            fixture.scheduler.last_epoch = 99
            random.seed(1)
            np.random.seed(2)
            torch.manual_seed(3)
            fixture.loader_generator.manual_seed(4)

            result = fixture.restore()
            actual_loader = collect_loader_trace(loader)
            actual_tail = (
                random.getrandbits(63),
                np.random.randint(0, 2**63 - 1, dtype=np.int64),
                torch.randint(0, 2**63 - 1, (5,), dtype=torch.int64),
            )

            self.assertEqual(result.epoch, 3)
            self.assertEqual(result.extra_state["skipped_singleton_batches"], 2)
            self.assertEqual(actual_loader, expected_loader)
            self.assertEqual(int(actual_tail[0]), int(expected_tail[0]))
            self.assertEqual(int(actual_tail[1]), int(expected_tail[1]))
            self.assertTrue(torch.equal(actual_tail[2], expected_tail[2]))
            self.assertEqual(fixture.scaler.scale, 512.0)
            self.assertEqual(fixture.scheduler.last_epoch, 1)
            self.assertAlmostEqual(
                fixture.optimizer.param_groups[0]["lr"],
                0.002,
            )
            get_cuda_state.assert_not_called()
            set_cuda_state.assert_not_called()

    def test_missing_required_state_is_rejected(self) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()
            cases: list[tuple[str, dict[str, object]]] = []
            missing_optimizer = copy.deepcopy(fixture.checkpoint)
            del missing_optimizer["optimizer"]
            cases.append(("optimizer", missing_optimizer))
            missing_identity = copy.deepcopy(fixture.checkpoint)
            del missing_identity["run_identity"]["run_id"]
            cases.append(("run_id", missing_identity))
            missing_python_rng = copy.deepcopy(fixture.checkpoint)
            del missing_python_rng["rng_state"]["python_random"]
            cases.append(("python_random", missing_python_rng))

            for label, checkpoint in cases:
                with self.subTest(label=label):
                    with self.assertRaises(exact.ExactResumeValidationError):
                        fixture.restore(checkpoint)

    def test_identity_epoch_and_metrics_boundary_mismatches_are_rejected(self) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()

            wrong_identity = dict(fixture.identity)
            wrong_identity["variant"] = "different_variant"
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "run identity mismatch",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=fixture.model,
                    optimizer=fixture.optimizer,
                    scaler=fixture.scaler,
                    scheduler=fixture.scheduler,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=wrong_identity,
                    expected_epoch=3,
                    expected_metrics_boundary=fixture.boundary,
                    expected_best_selection=fixture.selection,
                )

            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "epoch mismatch",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=fixture.model,
                    optimizer=fixture.optimizer,
                    scaler=fixture.scaler,
                    scheduler=fixture.scheduler,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=fixture.identity,
                    expected_epoch=2,
                    expected_metrics_boundary=fixture.boundary,
                    expected_best_selection=fixture.selection,
                )

            wrong_boundary = dict(fixture.boundary)
            wrong_boundary["metrics_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "metrics boundary mismatch",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=fixture.model,
                    optimizer=fixture.optimizer,
                    scaler=fixture.scaler,
                    scheduler=fixture.scheduler,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=fixture.identity,
                    expected_epoch=3,
                    expected_metrics_boundary=wrong_boundary,
                    expected_best_selection=fixture.selection,
                )

            tampered_selection = copy.deepcopy(fixture.checkpoint)
            tampered_selection["best_selection"]["primary"]["key"][0] = 0.95
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "best selection mismatch",
            ):
                fixture.restore(tampered_selection)

            internally_wrong = copy.deepcopy(fixture.checkpoint)
            internally_wrong["metrics_boundary"]["last_event_epoch"] = 2
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "last_event_epoch",
            ):
                fixture.restore(internally_wrong)

    def test_architecture_and_optional_scheduler_contracts_are_strict(self) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()
            different_model = nn.Linear(3, 2)
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "architecture/layout",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=different_model,
                    optimizer=fixture.optimizer,
                    scaler=fixture.scaler,
                    scheduler=fixture.scheduler,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=fixture.identity,
                    expected_epoch=3,
                    expected_metrics_boundary=fixture.boundary,
                    expected_best_selection=fixture.selection,
                )
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "requires a scheduler",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=fixture.model,
                    optimizer=fixture.optimizer,
                    scaler=fixture.scaler,
                    scheduler=None,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=fixture.identity,
                    expected_epoch=3,
                    expected_metrics_boundary=fixture.boundary,
                    expected_best_selection=fixture.selection,
                )

    def test_reversed_optimizer_parameter_order_is_rejected(self) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()
            reversed_optimizer = torch.optim.Adam(
                list(fixture.model.parameters())[::-1],
                lr=0.002,
            )
            reversed_scheduler = torch.optim.lr_scheduler.StepLR(
                reversed_optimizer,
                step_size=2,
                gamma=0.5,
            )
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "parameter name/order binding mismatch",
            ):
                exact.restore_exact_resume(
                    fixture.checkpoint,
                    model=fixture.model,
                    optimizer=reversed_optimizer,
                    scaler=StateScaler(),
                    scheduler=reversed_scheduler,
                    loader_generator=fixture.loader_generator,
                    expected_run_identity=fixture.identity,
                    expected_epoch=3,
                    expected_metrics_boundary=fixture.boundary,
                    expected_best_selection=fixture.selection,
                )

    def test_optimizer_binding_rejects_duplicate_omitted_and_external_params(
        self,
    ) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()
            parameters = list(fixture.model.parameters())
            omitted = torch.optim.Adam(parameters[:-1], lr=0.002)
            external_parameter = nn.Parameter(torch.zeros(1))
            external = torch.optim.Adam(
                [*parameters, external_parameter],
                lr=0.002,
            )
            duplicate = torch.optim.Adam(parameters, lr=0.002)
            duplicate.param_groups[0]["params"].append(parameters[0])
            cases = (
                ("missing model parameters", omitted),
                ("not a model parameter", external),
                ("appears more than once", duplicate),
            )
            for message, optimizer in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(
                        exact.ExactResumeValidationError,
                        message,
                    ):
                        exact.build_exact_resume_checkpoint(
                            model=fixture.model,
                            optimizer=optimizer,
                            scaler=StateScaler(),
                            epoch=3,
                            run_identity=fixture.identity,
                            best_selection=fixture.selection,
                            metrics_boundary=fixture.boundary,
                            loader_generator=fixture.loader_generator,
                        )

    def test_metrics_jsonl_boundary_hashes_contiguous_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [{"epoch": epoch, "loss": 1.0 / epoch} for epoch in range(1, 4)]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            boundary = exact.metrics_boundary_from_jsonl(path, expected_epoch=3)
            self.assertEqual(boundary["completed_epoch"], 3)
            self.assertEqual(boundary["event_count"], 3)
            self.assertEqual(boundary["last_event_epoch"], 3)
            self.assertEqual(len(boundary["metrics_sha256"]), 64)
            self.assertEqual(len(boundary["last_event_sha256"]), 64)

            path.write_text(
                json.dumps({"epoch": 1}) + "\n" + json.dumps({"epoch": 3}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "not contiguous",
            ):
                exact.metrics_boundary_from_jsonl(path, expected_epoch=2)

    def test_parent_warm_start_and_exact_resume_are_mutually_exclusive(self) -> None:
        with cpu_rng_environment():
            fixture = ExactResumeFixture()
            parent = exact.build_parent_warm_start_checkpoint(
                parent_model=fixture.model,
                parent_epoch=3,
                parent_identity=fixture.identity,
            )
            expected_weights = copy.deepcopy(fixture.model.state_dict())
            for parameter in fixture.model.parameters():
                parameter.data.zero_()
            result = exact.restore_parent_warm_start(
                parent,
                parent_model=fixture.model,
                expected_parent_identity=fixture.identity,
                expected_parent_epoch=3,
            )
            self.assertEqual(result.parent_epoch, 3)
            for name, value in fixture.model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[name]))

            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "mutually exclusive",
            ):
                exact.select_initialization_mode(
                    exact_resume="last.exact.pth",
                    parent_warm_start="parent.pth",
                )
            self.assertEqual(
                exact.select_initialization_mode(exact_resume="last.exact.pth"),
                exact.InitializationMode.EXACT_RESUME,
            )
            self.assertEqual(
                exact.select_initialization_mode(parent_warm_start="parent.pth"),
                exact.InitializationMode.PARENT_WARM_START,
            )
            with self.assertRaises(exact.ExactResumeValidationError):
                fixture.restore(parent)
            with self.assertRaises(exact.ExactResumeValidationError):
                exact.restore_parent_warm_start(
                    fixture.checkpoint,
                    parent_model=fixture.model,
                    expected_parent_identity=fixture.identity,
                )

    def test_atomic_save_preserves_previous_file_on_failure_then_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.exact.pth"
            exact.atomic_torch_save({"version": 1}, checkpoint)
            original_bytes = checkpoint.read_bytes()

            with mock.patch.object(
                exact.torch,
                "save",
                side_effect=OSError("injected save failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected save failure"):
                    exact.atomic_torch_save({"version": 2}, checkpoint)
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            self.assertFalse(
                any(
                    path.name.startswith(".last.exact.pth.tmp-")
                    for path in checkpoint.parent.iterdir()
                )
            )

            exact.atomic_torch_save({"version": 2}, checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["version"], 2)
            self.assertFalse(
                any(
                    path.name.startswith(".last.exact.pth.tmp-")
                    for path in checkpoint.parent.iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
