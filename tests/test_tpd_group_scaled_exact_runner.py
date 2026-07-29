from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import tpd_group_scaled_exact_runner as scaled_runner


BASE_LR = 0.004


def require(condition: bool, message: str = "") -> None:
    """Assertion helper that remains active under ``python -O``."""

    if not condition:
        raise AssertionError(message or "required condition is false")


class StateScaler:
    def __init__(self) -> None:
        self.updates = 0

    def state_dict(self) -> dict[str, int]:
        return {"updates": self.updates}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.updates = int(state["updates"])


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parent = nn.Linear(3, 5)
        self.qfg = nn.Linear(5, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.qfg(torch.tanh(self.parent(value)))


@contextmanager
def cpu_rng_environment():
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
        mock.patch.object(torch.cuda, "get_rng_state_all"),
        mock.patch.object(torch.cuda, "set_rng_state_all"),
    ):
        yield


def seed_all(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_components(
    *,
    scaled: bool = True,
    multipliers: tuple[float, float] = (0.1, 1.0),
) -> dict:
    seed_all()
    model = ToyModel()
    groups = [
        {"params": list(model.parent.parameters()), "lr": BASE_LR},
        {"params": list(model.qfg.parameters()), "lr": BASE_LR},
    ]
    if scaled:
        for group, name, multiplier in zip(
            groups,
            ("parent", "qfg"),
            multipliers,
        ):
            group["group_name"] = name
            group["schedule_multiplier"] = multiplier
    optimizer = torch.optim.Adam(groups, lr=BASE_LR, betas=(0.8, 0.95))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": generator,
    }


def make_spec(components: dict) -> exact_runner.ExactRunSpec:
    determinism = {
        "workers": 0,
        "explicit_loader_generator": True,
        "manual_lr": True,
    }
    if all(
        "schedule_multiplier" in group
        for group in components["optimizer"].param_groups
    ):
        determinism.update(
            scaled_runner.group_scaled_determinism_contract()
        )
    return exact_runner.ExactRunSpec(
        run_id="group-scaled-fixture",
        variant="group_scaled_fixture",
        dataset="fixture",
        seed=42,
        split_seed=3407,
        builder_metadata={"builder": "tests.ToyModel", "groups": 2},
        builder_manifest_sha256="1" * 64,
        source_locks={"model": "2" * 64, "training": "3" * 64},
        split_fingerprints={
            "train": exact_runner.OrderedFingerprint.from_values(
                "train", ("a", "b")
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation", ("c",)
            ),
        },
        data_fingerprints={
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples", ("a:image", "b:image")
            ),
            "validation_samples": exact_runner.OrderedFingerprint.from_values(
                "validation_samples", ("c:image",)
            ),
        },
        optimizer=exact_runner.optimizer_contract(
            components["model"], components["optimizer"]
        ),
        scaler=exact_runner.scaler_contract(
            components["scaler"], amp=False
        ),
        initialization_contract=exact_runner.fresh_initialization_contract(),
        lr_schedule=exact_runner.ManualCosineSchedule(
            total_epochs=3,
            base_lr=BASE_LR,
            min_lr=0.0001,
            warmup_epochs=1,
        ),
        loss={"name": "mse"},
        deep_supervision={"enabled": False, "outputs": 1},
        batch_size=2,
        patch_size=32,
        workers=0,
        amp=False,
        total_epochs=3,
        eval_interval=1,
        metric_config={"threshold": 0.5},
        environment={"device_type": "cpu", "torch": torch.__version__},
        determinism=determinism,
        initial_model_state_sha256=exact_runner.initial_model_state_sha256(
            components["model"]
        ),
        initial_rng=exact_runner.initial_rng_contract(),
        selection_policy=(
            exact_runner.pd_miou_selection_policy().normalized()
        ),
    )


def make_scaled_runner(root: Path, components: dict):
    return scaled_runner.GroupScaledExactRunner(
        root,
        model=components["model"],
        optimizer=components["optimizer"],
        scaler=components["scaler"],
        loader_generator=components["loader_generator"],
        spec=make_spec(components),
    )


def metrics(epoch: int, loss: float = 0.25) -> dict:
    return {
        "train_loss": loss,
        "pd": 0.7 + epoch * 0.01,
        "fa": 1e-5 / epoch,
        "tiny_pd": 0.6 + epoch * 0.01,
        "miou": 0.5 + epoch * 0.01,
        "val_loss": loss,
    }


def train_epoch(runner, components: dict) -> None:
    control = runner.next_epoch_control()
    inputs = torch.tensor(
        [[0.2, -0.1, 0.7], [0.4, 0.3, -0.2]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.1, 0.8], [0.6, -0.3]])
    optimizer = components["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    loss = (components["model"](inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    components["scaler"].updates += 1
    runner.commit_epoch(metrics(control.epoch, float(loss.detach())))


def state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def test_scaled_lr_evidence_and_checkpoint_lr_are_authoritative(tmp_path):
    with cpu_rng_environment():
        components = make_components()
        runner = make_scaled_runner(tmp_path / "run", components)
        runner.startup(exact_runner.InitializationRequest.fresh())
        control = runner.next_epoch_control()
        require(
            [g["lr"] for g in components["optimizer"].param_groups]
            == [control.learning_rate * 0.1, control.learning_rate]
        )
        runner.commit_epoch(metrics(1))

        event = json.loads(
            (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0]
        )
        require(event["optimizer_group_names"] == ["parent", "qfg"])
        require(event["schedule_multipliers"] == [0.1, 1.0])
        require(
            event["group_learning_rates"]
            == [control.learning_rate * 0.1, control.learning_rate]
        )
        require(
            [g["lr"] for g in components["optimizer"].param_groups]
            == [control.learning_rate, control.learning_rate]
        )
        payload = torch.load(
            runner.snapshot.active.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        require(
            [
                group["lr"]
                for group in payload["optimizer"]["state_dict"][
                    "param_groups"
                ]
            ]
            == [control.learning_rate, control.learning_rate]
        )


@pytest.mark.parametrize("bad_multiplier", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_multiplier_is_rejected(tmp_path, bad_multiplier):
    with cpu_rng_environment():
        components = make_components(multipliers=(bad_multiplier, 1.0))
        with pytest.raises(exact_runner.ExactRunnerError):
            make_scaled_runner(tmp_path / "bad", components)


def test_duplicate_group_name_and_caller_evidence_are_rejected(tmp_path):
    with cpu_rng_environment():
        components = make_components()
        components["optimizer"].param_groups[1]["group_name"] = "parent"
        with pytest.raises(exact_runner.ExactRunnerError, match="more than once"):
            make_scaled_runner(tmp_path / "duplicate", components)

        components = make_components()
        runner = make_scaled_runner(tmp_path / "forged", components)
        runner.startup(exact_runner.InitializationRequest.fresh())
        runner.next_epoch_control()
        with pytest.raises(exact_runner.ExactRunnerError, match="owned keys"):
            runner.commit_epoch(
                {**metrics(1), "group_learning_rates": [999.0, 999.0]}
            )


def test_changed_lr_or_multiplier_is_rejected(tmp_path):
    with cpu_rng_environment():
        components = make_components()
        runner = make_scaled_runner(tmp_path / "run", components)
        runner.startup(exact_runner.InitializationRequest.fresh())
        control = runner.next_epoch_control()
        components["optimizer"].param_groups[0]["lr"] = 9.0
        with pytest.raises(exact_runner.ExactRunnerError, match="LR differs"):
            runner.commit_epoch(metrics(1))
        components["optimizer"].param_groups[0]["lr"] = (
            control.learning_rate * 0.1
        )
        components["optimizer"].param_groups[0][
            "schedule_multiplier"
        ] = 0.2
        with pytest.raises(
            exact_runner.ExactRunnerError,
            match="schedule_multiplier changed",
        ):
            runner.commit_epoch(metrics(1))


def test_pre_pending_failure_restores_scaled_lr_for_corrected_retry(tmp_path):
    with cpu_rng_environment():
        components = make_components()
        runner = make_scaled_runner(tmp_path / "run", components)
        runner.startup(exact_runner.InitializationRequest.fresh())
        control = runner.next_epoch_control()
        with pytest.raises(exact_runner.ExactRunnerError, match="lacks"):
            runner.commit_epoch({"train_loss": 1.0})
        require(runner._pending is None)
        require(runner._open_control == control)
        require(
            [g["lr"] for g in components["optimizer"].param_groups]
            == [control.learning_rate * 0.1, control.learning_rate]
        )
        runner.commit_epoch(metrics(1))
        require(runner.snapshot.completed_epoch == 1)


def test_pending_write_failure_keeps_base_lr_and_retry_commits(tmp_path):
    with cpu_rng_environment():
        components = make_components()
        runner = make_scaled_runner(tmp_path / "run", components)
        runner.startup(exact_runner.InitializationRequest.fresh())
        control = runner.next_epoch_control()
        original_commit = runner.journal.commit
        with mock.patch.object(
            runner.journal,
            "commit",
            side_effect=RuntimeError("injected write failure"),
        ):
            with pytest.raises(RuntimeError, match="injected"):
                runner.commit_epoch(metrics(1))
        require(runner._pending is not None)
        require(
            [g["lr"] for g in components["optimizer"].param_groups]
            == [control.learning_rate, control.learning_rate]
        )
        with mock.patch.object(runner.journal, "commit", wraps=original_commit):
            runner.retry_pending_commit()
        require(runner.snapshot.completed_epoch == 1)


def test_all_one_multipliers_match_original_runner_updates(tmp_path):
    with cpu_rng_environment():
        base_components = make_components(scaled=False)
        base = exact_runner.ExactRunner(
            tmp_path / "base",
            model=base_components["model"],
            optimizer=base_components["optimizer"],
            scaler=base_components["scaler"],
            loader_generator=base_components["loader_generator"],
            spec=make_spec(base_components),
        )
        base.startup(exact_runner.InitializationRequest.fresh())
        for _ in range(3):
            train_epoch(base, base_components)

        scaled_components = make_components(multipliers=(1.0, 1.0))
        scaled = make_scaled_runner(tmp_path / "scaled", scaled_components)
        scaled.startup(exact_runner.InitializationRequest.fresh())
        for _ in range(3):
            train_epoch(scaled, scaled_components)

        require(
            state_equal(
                base_components["model"].state_dict(),
                scaled_components["model"].state_dict(),
            )
        )
        require(
            state_equal(
                base_components["optimizer"].state_dict()["state"],
                scaled_components["optimizer"].state_dict()["state"],
            )
        )


def test_exact_resume_reapplies_scaled_lr_and_matches_continuous(tmp_path):
    with cpu_rng_environment():
        continuous_components = make_components()
        continuous = make_scaled_runner(
            tmp_path / "continuous", continuous_components
        )
        continuous.startup(exact_runner.InitializationRequest.fresh())
        for _ in range(3):
            train_epoch(continuous, continuous_components)

        split_components = make_components()
        split = make_scaled_runner(tmp_path / "split", split_components)
        split.startup(exact_runner.InitializationRequest.fresh())
        train_epoch(split, split_components)

        rebuilt_components = make_components()
        rebuilt = make_scaled_runner(tmp_path / "split", rebuilt_components)
        snapshot = rebuilt.startup(exact_runner.InitializationRequest.exact())
        require(snapshot.completed_epoch == 1)
        require(
            [g["lr"] for g in rebuilt_components["optimizer"].param_groups]
            == [BASE_LR, BASE_LR]
        )
        control = rebuilt.next_epoch_control()
        require(
            [g["lr"] for g in rebuilt_components["optimizer"].param_groups]
            == [control.learning_rate * 0.1, control.learning_rate]
        )
        # Complete the already-open second epoch, then the third.
        inputs = torch.tensor([[0.2, -0.1, 0.7], [0.4, 0.3, -0.2]])
        targets = torch.tensor([[0.1, 0.8], [0.6, -0.3]])
        optimizer = rebuilt_components["optimizer"]
        optimizer.zero_grad(set_to_none=True)
        loss = (rebuilt_components["model"](inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        rebuilt_components["scaler"].updates += 1
        rebuilt.commit_epoch(metrics(2, float(loss.detach())))
        train_epoch(rebuilt, rebuilt_components)
        require(
            state_equal(
                continuous_components["model"].state_dict(),
                rebuilt_components["model"].state_dict(),
            )
        )
        require(
            state_equal(
                continuous_components["optimizer"].state_dict(),
                rebuilt_components["optimizer"].state_dict(),
            )
        )
        require(
            (tmp_path / "continuous" / "metrics.jsonl").read_bytes()
            == (tmp_path / "split" / "metrics.jsonl").read_bytes()
        )


def test_freeze_batchnorm_keeps_buffers_and_affine_gradients():
    seed_all()
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
    )
    model.train()
    batchnorm = model[1]
    before = {
        "mean": batchnorm.running_mean.clone(),
        "var": batchnorm.running_var.clone(),
        "count": batchnorm.num_batches_tracked.clone(),
    }
    require(scaled_runner.freeze_batchnorm_running_stats(model) == 1)
    require(not batchnorm.training)
    require(batchnorm.weight.requires_grad)
    require(batchnorm.bias.requires_grad)
    model(torch.randn(3, 3, 8, 8)).square().mean().backward()
    require(torch.equal(batchnorm.running_mean, before["mean"]))
    require(torch.equal(batchnorm.running_var, before["var"]))
    require(torch.equal(batchnorm.num_batches_tracked, before["count"]))
    require(batchnorm.weight.grad is not None)
    require(batchnorm.bias.grad is not None)
