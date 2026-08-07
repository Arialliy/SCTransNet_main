from __future__ import annotations

import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis import train_nuaa_current_scalar_calibration_control_v1 as subject


torch.set_num_threads(1)


class _ToyInternalDataset(Dataset):
    def __init__(self) -> None:
        self.epoch = 0
        self.images = torch.tensor(
            [
                [[[0.2, -0.5], [0.8, -0.1]]],
                [[[-0.3, 0.6], [0.4, -0.7]]],
            ],
            dtype=torch.float32,
        )
        self.masks = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.masks[index]


class _ToyWarmStartShell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.pbdr_v3 = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.base.weight.fill_(1.25)
            self.pbdr_v3.weight.fill_(0.5)
            self.pbdr_v3.bias.fill_(0.1)

    def forward_for_pbdr_v3_training(self, images: torch.Tensor):
        base_logits = self.bn(self.base(images))
        return (), SimpleNamespace(base_logits=base_logits)


def _point(
    threshold: float,
    *,
    matched: int,
    fa: float,
    miou: float,
    niou: float,
):
    return {
        "threshold": threshold,
        "matched_target_count": matched,
        "target_count": 10,
        "fa": fa,
        "miou": miou,
        "niou": niou,
    }


def _valid_split() -> dict:
    official = [f"nuaa_train_{index:03d}" for index in range(213)]
    payload = {
        "schema": "sctransnet_nuaa_pbdr_v3_internal_split/v1",
        "dataset": subject.DATASET,
        "source_split": "official_train_only",
        "official_test_index_opened": False,
        "split_seed": subject.SPLIT_SEED,
        "val_fraction": subject.VALIDATION_FRACTION,
        "official_train_ids": official,
        "development_train_ids": official[:170],
        "internal_validation_ids": official[170:],
        "mask_stats": [],
        "official_train_index_sha256": (
            subject.data_protocol.EXPECTED_SPLITS[subject.DATASET]["train"][
                "file_sha256"
            ]
        ),
        "data_protocol_manifest": {
            "path": "/immutable/protocol.json",
            "sha256": "f" * 64,
        },
    }
    payload["split_sha256"] = subject.v3_registry.canonical_sha256(payload)
    return payload


class NUAACurrentScalarCalibrationControlTests(unittest.TestCase):
    def test_identity_initialization_is_exact_and_has_two_gradients(self) -> None:
        calibrator = subject.ScalarTemperatureBiasCalibrator()
        logits = torch.tensor([[[[-1.2, 0.3], [0.8, 2.0]]]], requires_grad=True)
        output = calibrator(logits)
        self.assertTrue(torch.equal(output, logits))
        self.assertEqual(sum(value.numel() for value in calibrator.parameters()), 2)
        self.assertEqual(
            tuple(calibrator.state_dict()), ("log_temperature", "bias")
        )
        output.square().sum().backward()
        self.assertIsNotNone(calibrator.log_temperature.grad)
        self.assertIsNotNone(calibrator.bias.grad)
        self.assertTrue(torch.isfinite(calibrator.log_temperature.grad))
        self.assertTrue(torch.isfinite(calibrator.bias.grad))

    def test_shell_and_pbdr_v3_are_frozen_while_only_scalars_train(self) -> None:
        model = _ToyWarmStartShell()
        audit = subject.freeze_v3_shell_for_scalar_control(model)
        self.assertEqual(audit["shell_trainable_parameter_count"], 0)
        self.assertTrue(audit["pbdr_v3_frozen"])
        self.assertFalse(audit["shell_training"])
        self.assertFalse(audit["pbdr_v3_training"])
        self.assertFalse(any(value.requires_grad for value in model.parameters()))
        self.assertFalse(any(module.training for module in model.modules()))

        calibrator = subject.ScalarTemperatureBiasCalibrator()
        optimizer = subject.build_optimizer(calibrator)
        optimized = sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        self.assertEqual(optimized, 2)

    def test_one_epoch_changes_scalars_but_not_warm_start_shell(self) -> None:
        model = _ToyWarmStartShell()
        freeze = subject.freeze_v3_shell_for_scalar_control(model)
        shell_before = freeze["full_shell_state_sha256"]
        calibrator = subject.ScalarTemperatureBiasCalibrator()
        optimizer = subject.build_optimizer(calibrator)
        dataset = _ToyInternalDataset()
        result = subject.train_scalar_epoch(
            model=model,
            calibrator=calibrator,
            loader=DataLoader(dataset, batch_size=2, shuffle=False),
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=1,
        )
        self.assertEqual(dataset.epoch, 1)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["batch_count"], 1)
        self.assertTrue(np.isfinite(result["mean_bce_with_logits"]))
        self.assertNotEqual(
            (
                float(calibrator.log_temperature.detach()),
                float(calibrator.bias.detach()),
            ),
            (0.0, 0.0),
        )
        shell_after = subject.v3_registry.tensor_mapping_sha256(model.state_dict())
        self.assertEqual(shell_after, shell_before)
        self.assertTrue(all(value.grad is None for value in model.parameters()))

    def test_threshold_grid_and_validation_selector_prefer_passing_gate(self) -> None:
        grid = subject.build_threshold_grid()
        self.assertEqual(len(grid), 61)
        self.assertEqual(grid[0], 0.2)
        self.assertEqual(grid[-1], 0.8)
        self.assertIn(0.5, grid)
        current = _point(0.5, matched=9, fa=2.0e-5, miou=0.80, niou=0.79)
        points = [
            _point(0.4, matched=10, fa=3.0e-5, miou=0.83, niou=0.81),
            _point(0.5, matched=9, fa=1.9e-5, miou=0.801, niou=0.80),
            _point(0.6, matched=9, fa=1.8e-5, miou=0.803, niou=0.80),
        ]
        selected = subject.select_validation_point(points, current)
        self.assertEqual(selected["passing_point_count"], 1)
        self.assertEqual(selected["selected"]["threshold"], 0.6)
        self.assertTrue(selected["selected"]["certification"]["passed"])

    def test_identity_binds_parent_sources_split_and_100_epoch_cap(self) -> None:
        parent = {
            "checkpoint_role": "best_miou",
            "sha256": "a" * 64,
            "path": "/immutable/current.pth.tar",
            "state_sha256": "b" * 64,
        }
        split = _valid_split()
        sources = {
            "source.py": {
                "path": "/repo/source.py",
                "sha256": "c" * 64,
                "bytes": 10,
            }
        }
        identity = subject.build_run_identity(
            parent_role="best_miou",
            parent_record=parent,
            split_manifest=split,
            source_records=sources,
            epochs=100,
        )
        self.assertEqual(identity["maximum_epochs"], 100)
        self.assertFalse(identity["official_test_access_authorized"])
        changed_split = dict(split)
        changed_split["development_train_ids"] = list(
            split["development_train_ids"]
        )
        changed_split["internal_validation_ids"] = list(
            split["internal_validation_ids"]
        )
        changed_split["development_train_ids"][0], changed_split[
            "internal_validation_ids"
        ][0] = (
            changed_split["internal_validation_ids"][0],
            changed_split["development_train_ids"][0],
        )
        changed_split["split_sha256"] = subject.v3_registry.canonical_sha256(
            {
                key: value
                for key, value in changed_split.items()
                if key != "split_sha256"
            }
        )
        changed = subject.build_run_identity(
            parent_role="best_miou",
            parent_record=parent,
            split_manifest=changed_split,
            source_records=sources,
            epochs=100,
        )
        self.assertNotEqual(identity["identity_sha256"], changed["identity_sha256"])
        with self.assertRaisesRegex(subject.ScalarCalibrationProtocolError, "epochs"):
            subject.build_run_identity(
                parent_role="best_miou",
                parent_record=parent,
                split_manifest=split,
                source_records=sources,
                epochs=101,
            )

    def test_split_manifest_rejects_any_official_test_access(self) -> None:
        split = _valid_split()
        split["official_test_index_opened"] = True
        split["split_sha256"] = subject.v3_registry.canonical_sha256(
            {
                key: value
                for key, value in split.items()
                if key != "split_sha256"
            }
        )
        with self.assertRaisesRegex(
            subject.ScalarCalibrationProtocolError, "official test"
        ):
            subject.validate_internal_split_manifest(split)

    def test_cached_validation_reports_fixed_sweep_and_empty_control(self) -> None:
        cache = {
            "base_logits": [
                np.array([[2.0, -2.0], [-1.0, 1.0]], dtype=np.float32)
            ],
            "targets": [
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            ],
        }
        evaluated = subject.evaluate_scalar_values_on_cache(
            cache=cache,
            log_temperature=0.0,
            bias=0.0,
            threshold_grid=(0.2, 0.5, 0.8),
        )
        self.assertEqual(
            [point["threshold"] for point in evaluated["registered_points"]],
            [0.2, 0.5, 0.8],
        )
        self.assertEqual(
            evaluated["fixed_threshold_0_5"],
            evaluated["registered_points"][1],
        )
        endpoint = evaluated["threshold_1_0_empty_control"]
        self.assertEqual(endpoint["threshold"], 1.0)
        self.assertEqual(endpoint["matched_target_count"], 0)
        self.assertEqual(endpoint["fa"], 0.0)
        self.assertFalse(evaluated["probability_cache_written"])

    def test_resume_rejects_any_identity_or_scalar_state_drift(self) -> None:
        calibrator = subject.ScalarTemperatureBiasCalibrator()
        optimizer = subject.build_optimizer(calibrator)
        payload = {
            "schema": subject.RESUME_SCHEMA,
            "identity_sha256": "d" * 64,
            "completed_epoch": 4,
            "calibrator_state": calibrator.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": subject.capture_rng_state(),
            "history": [{"epoch": epoch} for epoch in range(1, 5)],
            "selection": {},
        }
        subject.validate_resume_payload(
            payload,
            expected_identity_sha256="d" * 64,
            maximum_completed_epoch=100,
        )
        bad_identity = dict(payload, identity_sha256="e" * 64)
        with self.assertRaisesRegex(
            subject.ScalarCalibrationProtocolError,
            "source/parent/split",
        ):
            subject.validate_resume_payload(
                bad_identity,
                expected_identity_sha256="d" * 64,
                maximum_completed_epoch=100,
            )
        bad_state = dict(payload, calibrator_state={"bias": torch.zeros(())})
        with self.assertRaisesRegex(
            subject.ScalarCalibrationProtocolError, "state keys"
        ):
            subject.validate_resume_payload(
                bad_state,
                expected_identity_sha256="d" * 64,
                maximum_completed_epoch=100,
            )
        bad_history = dict(payload, history=payload["history"][:-1])
        with self.assertRaisesRegex(
            subject.ScalarCalibrationProtocolError, "history length"
        ):
            subject.validate_resume_payload(
                bad_history,
                expected_identity_sha256="d" * 64,
                maximum_completed_epoch=100,
            )

    def test_rng_capture_restore_is_exact(self) -> None:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        state = subject.capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        random.random()
        np.random.rand()
        torch.rand(3)
        subject.restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_tf32_is_explicitly_disabled(self) -> None:
        contract = subject.configure_inference_math()
        self.assertFalse(contract["cuda_matmul_allow_tf32"])
        self.assertFalse(contract["cudnn_allow_tf32"])
        self.assertEqual(contract["float32_matmul_precision"], "highest")
        self.assertTrue(contract["deterministic_algorithms"])


if __name__ == "__main__":
    unittest.main()
