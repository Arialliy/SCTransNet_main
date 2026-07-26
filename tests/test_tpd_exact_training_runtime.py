from __future__ import annotations

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_epoch_journal as journal_module
from experiments import tpd_exact_resume as exact
from experiments import tpd_exact_training_runtime as runtime_module


class StateScaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


@contextmanager
def cpu_rng_environment():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
        mock.patch.object(torch.cuda, "get_rng_state_all"),
        mock.patch.object(torch.cuda, "set_rng_state_all"),
    ):
        yield


def make_components(seed: int = 7) -> dict:
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(991 + seed)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": StateScaler(512.0),
        "loader_generator": loader_generator,
    }


def optimizer_step(components: dict) -> None:
    model = components["model"]
    optimizer = components["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]])
    model(inputs).square().mean().backward()
    optimizer.step()
    components["scheduler"].step()


IDENTITY = {
    "run_id": "runtime-fixture",
    "variant": "tpd_fixture",
    "architecture_id": "tpd_fixture_v1",
    "dataset": "fixture",
    "seed": 42,
    "split_seed": 3407,
    "split_sha256": "a" * 64,
    "source_lock_sha256": "b" * 64,
}


def best_selection(epoch: int) -> dict:
    metrics = {
        "pd": 0.9 + 0.001 * epoch,
        "fa": 1.0e-6,
        "miou": 0.8 + 0.001 * epoch,
        "val_loss": 0.2,
        "tiny_pd": 1.0,
        "tiny_target_count": 3,
    }
    return {
        "primary": {
            "role": "best_validation_pd_primary",
            "epoch": epoch,
            "key": [metrics["pd"], -metrics["fa"], metrics["miou"]],
            "metrics": metrics,
        },
        "secondary": {
            "role": "best_validation_miou_secondary",
            "epoch": epoch,
            "key": [metrics["miou"], metrics["pd"], -metrics["fa"]],
            "metrics": metrics,
        },
    }


def make_runtime(
    root: Path,
    components: dict,
) -> runtime_module.ExactTrainingRuntime:
    return runtime_module.ExactTrainingRuntime(
        root,
        model=components["model"],
        optimizer=components["optimizer"],
        scaler=components["scaler"],
        scheduler=components["scheduler"],
        loader_generator=components["loader_generator"],
    )


def commit_epoch(
    runtime: runtime_module.ExactTrainingRuntime,
    components: dict,
    epoch: int,
) -> runtime_module.ExactTrainingSnapshot:
    optimizer_step(components)
    pending = runtime.prepare_epoch(
        {"epoch": epoch, "loss": 1.0 / epoch},
        best_selection=best_selection(epoch),
        extra_state={"skipped_singleton_batches": epoch},
    )
    return runtime.commit_epoch(pending)


class ExactTrainingRuntimeTest(unittest.TestCase):
    def test_fresh_start_and_contiguous_epoch_commits(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            components = make_components()
            runtime = make_runtime(Path(directory) / "journal", components)
            started = runtime.startup(
                exact.InitializationMode.FRESH,
                run_identity=IDENTITY,
            )
            self.assertEqual(started.completed_epoch, 0)
            first = commit_epoch(runtime, components, 1)
            second = commit_epoch(runtime, components, 2)
            self.assertEqual(first.completed_epoch, 1)
            self.assertEqual(second.completed_epoch, 2)
            self.assertEqual(second.active.slot, "slot_b")
            self.assertEqual(
                second.metrics_boundary["completed_epoch"],
                2,
            )
            with self.assertRaisesRegex(
                runtime_module.ExactTrainingRuntimeError,
                "startup may be called only once",
            ):
                runtime.startup(
                    exact.InitializationMode.FRESH,
                    run_identity=IDENTITY,
                )

    def test_process_reconstruction_restores_and_continues(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "journal"
            source_components = make_components(7)
            source = make_runtime(root, source_components)
            source.startup("fresh", run_identity=IDENTITY)
            commit_epoch(source, source_components, 1)
            committed = commit_epoch(source, source_components, 2)
            expected_weights = copy.deepcopy(
                source_components["model"].state_dict()
            )

            resumed_components = make_components(1234)
            resumed = make_runtime(root, resumed_components)
            restored = resumed.startup(
                "exact_resume",
                run_identity=IDENTITY,
                expected_epoch=2,
                expected_metrics_boundary=committed.metrics_boundary,
                expected_best_selection=committed.best_selection,
            )
            self.assertEqual(restored.completed_epoch, 2)
            for name, value in resumed_components["model"].state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[name]))
            third = commit_epoch(resumed, resumed_components, 3)
            self.assertEqual(third.completed_epoch, 3)
            self.assertEqual(third.active.slot, "slot_a")

    def test_external_boundary_identity_and_best_mismatch_are_rejected(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "journal"
            components = make_components()
            writer = make_runtime(root, components)
            writer.startup("fresh", run_identity=IDENTITY)
            committed = commit_epoch(writer, components, 1)

            wrong_boundary = copy.deepcopy(committed.metrics_boundary)
            wrong_boundary["metrics_sha256"] = "0" * 64
            cases = []
            cases.append(
                (
                    "boundary differs",
                    IDENTITY,
                    wrong_boundary,
                    committed.best_selection,
                )
            )
            wrong_identity = copy.deepcopy(IDENTITY)
            wrong_identity["run_id"] = "another-run"
            cases.append(
                (
                    "run identity mismatch",
                    wrong_identity,
                    committed.metrics_boundary,
                    committed.best_selection,
                )
            )
            wrong_best = copy.deepcopy(committed.best_selection)
            wrong_best["primary"]["metrics"]["pd"] = 0.1
            cases.append(
                (
                    "best selection mismatch",
                    IDENTITY,
                    committed.metrics_boundary,
                    wrong_best,
                )
            )
            for message, identity, boundary, selection in cases:
                with self.subTest(message=message):
                    fresh_components = make_components(99)
                    reader = make_runtime(root, fresh_components)
                    with self.assertRaisesRegex(Exception, message):
                        reader.startup(
                            "exact_resume",
                            run_identity=identity,
                            expected_epoch=1,
                            expected_metrics_boundary=boundary,
                            expected_best_selection=selection,
                        )

    def test_marker_failure_does_not_advance_runtime_or_journal(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "journal"
            components = make_components()
            runtime = make_runtime(root, components)
            runtime.startup("fresh", run_identity=IDENTITY)
            commit_epoch(runtime, components, 1)
            optimizer_step(components)
            pending = runtime.prepare_epoch(
                {"epoch": 2, "loss": 0.5},
                best_selection=best_selection(2),
            )
            real_atomic_write = journal_module._atomic_write_bytes

            def fail_marker(destination: Path, content: bytes) -> None:
                if destination.name == journal_module.MARKER_FILENAME:
                    raise OSError("injected marker failure")
                real_atomic_write(destination, content)

            with mock.patch.object(
                journal_module,
                "_atomic_write_bytes",
                side_effect=fail_marker,
            ):
                with self.assertRaisesRegex(OSError, "marker failure"):
                    runtime.commit_epoch(pending)
            self.assertEqual(runtime.snapshot.completed_epoch, 1)
            on_disk = journal_module.ExactEpochJournal(root).load_active()
            self.assertIsNotNone(on_disk)
            self.assertEqual(on_disk.epoch, 1)

            retried = runtime.commit_epoch(pending)
            self.assertEqual(retried.completed_epoch, 2)

    def test_startup_mode_and_empty_active_contracts(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "journal"
            components = make_components()
            runtime = make_runtime(root, components)
            with self.assertRaisesRegex(
                runtime_module.ExactTrainingRuntimeError,
                "requires a committed journal",
            ):
                runtime.startup(
                    "exact_resume",
                    run_identity=IDENTITY,
                    expected_epoch=1,
                    expected_metrics_boundary={
                        "completed_epoch": 1,
                        "event_count": 1,
                        "last_event_epoch": 1,
                        "metrics_sha256": "a" * 64,
                        "last_event_sha256": "b" * 64,
                    },
                    expected_best_selection=best_selection(1),
                )

            writer_components = make_components()
            writer = make_runtime(root, writer_components)
            writer.startup("fresh", run_identity=IDENTITY)
            commit_epoch(writer, writer_components, 1)
            new_components = make_components()
            fresh_again = make_runtime(root, new_components)
            with self.assertRaisesRegex(
                runtime_module.ExactTrainingRuntimeError,
                "requires an empty journal",
            ):
                fresh_again.startup("fresh", run_identity=IDENTITY)


if __name__ == "__main__":
    unittest.main()
