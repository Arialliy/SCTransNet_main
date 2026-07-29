from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import tpd_exact_resume as exact_resume
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as entry,
)


class StateScaler:
    def __init__(self, updates: int = 0) -> None:
        self.updates = int(updates)

    def state_dict(self) -> dict[str, int]:
        return {"updates": self.updates}

    def load_state_dict(self, state: dict[str, int]) -> None:
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


def provenance() -> dict:
    return {
        "schema": entry.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(
            entry.PARENT_CHECKPOINT_PATH.resolve()
        ),
        "parent_checkpoint_sha256": entry.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": ["state_dict"],
        "parent_state_key_count": 544,
        "preserved_new_state_key_count": 4,
        "new_module_prefixes": ["target_survival"],
        "zero_init_prefixes": [
            "target_survival.heads.emb1.classifier",
            "target_survival.heads.emb2.classifier",
        ],
    }


def make_components() -> dict:
    seed_all(entry.TRAINING_SEED)
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Tanh(),
        nn.Linear(5, 2),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=entry.FORMAL_BASE_LR,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(entry.TRAINING_SEED)
    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": StateScaler(),
        "loader_generator": loader_generator,
    }


def make_spec(
    components: dict,
    *,
    initial_rng: dict | None = None,
) -> exact_runner.ExactRunSpec:
    model = components["model"]
    initial_sha = exact_runner.initial_model_state_sha256(model)
    initialization = exact_runner.extension_parent_initialization_contract(
        provenance(),
        loaded_child_model_state_sha256=initial_sha,
    )
    statistics = entry.load_survival_target_statistics()
    selection = exact_runner.pd_miou_selection_policy()
    return exact_runner.ExactRunSpec(
        run_id=(
            f"{entry.RUN_ID_PREFIX}NUDT-SIRST:"
            f"{entry.TSS_CONTROL_VARIANT}:resume-fixture"
        ),
        variant=entry.TSS_CONTROL_VARIANT,
        dataset="NUDT-SIRST",
        seed=entry.TRAINING_SEED,
        split_seed=entry.SPLIT_SEED,
        builder_metadata={
            "schema": "tiny_tss_resume_builder_v1",
            "state_layout": "linear_3_5_2",
        },
        builder_manifest_sha256="a" * 64,
        source_locks={
            entry.SOURCE_LOCK_KEY: "1" * 64,
            "training_data": "2" * 64,
            "survival_target_statistics": statistics["sha256"],
            "parent_checkpoint": entry.PARENT_CHECKPOINT_SHA256,
        },
        split_fingerprints={
            "train": exact_runner.OrderedFingerprint.from_values(
                "train",
                ("a", "b", "c", "d"),
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation",
                ("e",),
            ),
        },
        data_fingerprints={
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples",
                ("a:0", "b:1", "c:2", "d:3"),
            ),
        },
        optimizer=exact_runner.optimizer_contract(
            model,
            components["optimizer"],
        ),
        scaler=exact_runner.scaler_contract(
            components["scaler"],
            amp=False,
        ),
        initialization_contract=initialization,
        lr_schedule=exact_runner.ManualCosineSchedule(
            total_epochs=3,
            base_lr=entry.FORMAL_BASE_LR,
            min_lr=entry.FORMAL_MIN_LR,
            warmup_epochs=1,
        ),
        loss=entry._loss_contract(
            entry.TSS_CONTROL_VARIANT,
            statistics,
        ),
        deep_supervision={
            "enabled": True,
            "training_output": "TPDForwardOutput",
            "validation_output": "legacy",
        },
        batch_size=2,
        patch_size=entry.FORMAL_PATCH_SIZE,
        workers=0,
        amp=False,
        total_epochs=3,
        eval_interval=1,
        metric_config={
            "threshold": entry.FORMAL_THRESHOLD,
            "match_radius": entry.FORMAL_MATCH_RADIUS,
            "tiny_area": entry.FORMAL_TINY_AREA,
            "official_test_accessed": False,
        },
        environment={"name": "cpu-resume-fixture"},
        determinism=entry._required_determinism(
            entry.TSS_CONTROL_VARIANT,
            statistics,
        ),
        initial_model_state_sha256=initial_sha,
        initial_rng=(
            copy.deepcopy(initial_rng)
            if initial_rng is not None
            else exact_runner.initial_rng_contract()
        ),
        selection_policy=selection.normalized(),
    )


def make_runner(
    root: Path,
    components: dict,
    spec: exact_runner.ExactRunSpec,
) -> entry.TPDNERV8V4SurvivalExactRunner:
    return entry.TPDNERV8V4SurvivalExactRunner(
        root,
        model=components["model"],
        optimizer=components["optimizer"],
        scaler=components["scaler"],
        loader_generator=components["loader_generator"],
        spec=spec,
        selection_policy=exact_runner.pd_miou_selection_policy(),
    )


def run_one_epoch(
    runner: entry.TPDNERV8V4SurvivalExactRunner,
    components: dict,
) -> dict:
    control = runner.next_epoch_control()
    permutation = torch.randperm(
        8,
        generator=components["loader_generator"],
    )
    python_trace = random.getrandbits(31)
    numpy_trace = int(np.random.randint(0, 2**31 - 1))
    inputs = torch.rand(2, 3)
    target = torch.tensor(
        [
            [
                (python_trace % 101) / 101.0,
                (numpy_trace % 103) / 103.0,
            ],
            [
                (numpy_trace % 107) / 107.0,
                (python_trace % 109) / 109.0,
            ],
        ],
        dtype=torch.float32,
    )
    optimizer = components["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    loss = (components["model"](inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    components["scaler"].updates += 1
    fields = {
        "train_total_loss": float(loss.detach().item()),
        "train_segmentation_loss": float(loss.detach().item()),
        "train_survival_loss": 0.0,
        "train_survival_emb1_loss": None,
        "train_survival_emb2_loss": None,
        "loader_trace": permutation.tolist(),
        "python_trace": python_trace,
        "numpy_trace": numpy_trace,
        "pd": 0.70 + 0.01 * control.epoch,
        "fa": 1e-5 / control.epoch,
        "tiny_pd": 0.60 + 0.01 * control.epoch,
        "miou": 0.50 + 0.02 * control.epoch,
        "val_loss": float(loss.detach().item()),
    }
    snapshot = runner.commit_epoch(
        fields,
        extra_state={
            "variant": entry.TSS_CONTROL_VARIANT,
            "survival_weight": 0.0,
            "scaler_updates": components["scaler"].updates,
        },
    )
    return {
        "epoch": control.epoch,
        "learning_rate": control.learning_rate,
        "permutation": permutation.tolist(),
        "python": python_trace,
        "numpy": numpy_trace,
        "loss": float(loss.detach().item()),
        "selection": copy.deepcopy(snapshot.best_selection),
    }


def nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)

class V4SurvivalExactResumeTests(unittest.TestCase):
    def test_continuous_equals_epoch_boundary_exact_resume(self) -> None:
        with cpu_rng_environment(), tempfile.TemporaryDirectory() as text:
            root = Path(text)

            continuous_components = make_components()
            initial_rng = exact_runner.initial_rng_contract()
            continuous_spec = make_spec(
                continuous_components,
                initial_rng=initial_rng,
            )
            continuous_runner = make_runner(
                root / "continuous",
                continuous_components,
                continuous_spec,
            )
            initial_sha = exact_runner.initial_model_state_sha256(
                continuous_components["model"]
            )
            parent_request = (
                exact_runner.InitializationRequest.extension_parent(
                    provenance(),
                    loaded_child_model_state_sha256=initial_sha,
                )
            )
            started = continuous_runner.startup(parent_request)
            self.assertEqual(started.completed_epoch, 0)
            self.assertEqual(
                started.initialization_mode,
                exact_resume.InitializationMode.PARENT_WARM_START,
            )
            self.assertEqual(
                started.parent_provenance["mode"],
                exact_runner.EXTENSION_PARENT_MODE,
            )
            continuous_trace = [
                run_one_epoch(continuous_runner, continuous_components)
                for _ in range(3)
            ]

            split_components = make_components()
            split_spec = make_spec(
                split_components,
                initial_rng=initial_rng,
            )
            split_runner = make_runner(
                root / "split",
                split_components,
                split_spec,
            )
            split_runner.startup(parent_request)
            split_trace = [
                run_one_epoch(split_runner, split_components)
            ]

            # Rebuild all mutable objects as a new process would.  Startup must
            # restore model, optimizer, scaler, global RNG, loader RNG,
            # selection state, and parent provenance from the journal.
            resumed_components = make_components()
            resumed_spec = make_spec(
                resumed_components,
                initial_rng=initial_rng,
            )
            seed_all(999)
            resumed_runner = make_runner(
                root / "split",
                resumed_components,
                resumed_spec,
            )
            resumed = resumed_runner.startup(
                exact_runner.InitializationRequest.exact()
            )
            self.assertEqual(resumed.completed_epoch, 1)
            self.assertEqual(
                resumed.parent_provenance["mode"],
                exact_runner.EXTENSION_PARENT_MODE,
            )
            split_trace.extend(
                run_one_epoch(resumed_runner, resumed_components)
                for _ in range(2)
            )

            self.assertEqual(continuous_trace, split_trace)
            self.assertTrue(
                nested_equal(
                    continuous_components["model"].state_dict(),
                    resumed_components["model"].state_dict(),
                )
            )
            self.assertTrue(
                nested_equal(
                    continuous_components["optimizer"].state_dict(),
                    resumed_components["optimizer"].state_dict(),
                )
            )
            self.assertEqual(
                continuous_components["scaler"].state_dict(),
                resumed_components["scaler"].state_dict(),
            )
            self.assertTrue(
                torch.equal(
                    continuous_components[
                        "loader_generator"
                    ].get_state(),
                    resumed_components["loader_generator"].get_state(),
                )
            )
            continuous_events = (
                root / "continuous" / exact_runner.METRICS_FILENAME
            ).read_text(encoding="utf-8")
            split_events = (
                root / "split" / exact_runner.METRICS_FILENAME
            ).read_text(encoding="utf-8")
            self.assertEqual(continuous_events, split_events)

            continuous_active = continuous_runner.snapshot.active
            resumed_active = resumed_runner.snapshot.active
            self.assertIsNotNone(continuous_active)
            self.assertIsNotNone(resumed_active)
            continuous_payload = torch.load(
                continuous_active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            resumed_payload = torch.load(
                resumed_active.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertTrue(
                nested_equal(continuous_payload, resumed_payload)
            )

    def test_cross_variant_resume_is_rejected_before_restore(self) -> None:
        components = make_components()
        spec = make_spec(components)
        identity = exact_runner.build_run_identity(
            components["model"],
            spec,
        )
        wrong = copy.deepcopy(identity)
        wrong["variant"] = entry.TSS_ON_VARIANT
        wrong["training_contract"]["loss"] = entry._loss_contract(
            entry.TSS_ON_VARIANT,
            entry.load_survival_target_statistics(),
        )
        wrong["training_contract"]["determinism"] = (
            entry._required_determinism(
                entry.TSS_ON_VARIANT,
                entry.load_survival_target_statistics(),
            )
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            entry.require_tss_run_identity(
                wrong,
                label="wrong variant",
                expected_variant=entry.TSS_CONTROL_VARIANT,
            )


if __name__ == "__main__":
    unittest.main()
