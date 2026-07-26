from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from experiments import train_tpd_pilot as base
from experiments.tpd_ner_runtime import (
    atomic_torch_save,
    checked_deep_supervision_loss,
    checked_validate,
    guarded_training_runtime,
)


class TPDNERRuntimeTests(unittest.TestCase):
    def test_checked_loss_rejects_non_finite_objective(self) -> None:
        outputs = (torch.full((2, 1, 4, 4), float("nan")),)
        targets = torch.zeros(2, 1, 4, 4)
        with self.assertRaisesRegex(FloatingPointError, "loss is non-finite"):
            checked_deep_supervision_loss(outputs, targets, nn.MSELoss())

    def test_atomic_checkpoint_replacement_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pth.tar"
            atomic_torch_save({"value": torch.tensor([1.0, 2.0])}, checkpoint)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            torch.testing.assert_close(loaded["value"], torch.tensor([1.0, 2.0]))
            self.assertFalse(
                any(
                    path.name.startswith(".best.pth.tar.tmp-")
                    for path in checkpoint.parent.iterdir()
                )
            )

    def test_validation_guard_allows_only_no_tiny_target_nan(self) -> None:
        with mock.patch(
            "experiments.tpd_ner_runtime._ORIGINAL_VALIDATE",
            return_value={
                "tiny_pd": float("nan"),
                "tiny_target_count": 0,
                "miou": 0.5,
            },
        ):
            metrics = checked_validate()
            self.assertEqual(metrics["tiny_target_count"], 0)
        with mock.patch(
            "experiments.tpd_ner_runtime._ORIGINAL_VALIDATE",
            return_value={
                "tiny_pd": float("nan"),
                "tiny_target_count": 1,
                "miou": 0.5,
            },
        ):
            with self.assertRaisesRegex(FloatingPointError, "tiny_pd"):
                checked_validate()

    def test_runtime_guards_are_scoped_and_restored(self) -> None:
        original_loss = base.deep_supervision_loss
        original_validate = base.validate
        original_save = torch.save
        with guarded_training_runtime():
            self.assertIs(base.deep_supervision_loss, checked_deep_supervision_loss)
            self.assertIs(torch.save, atomic_torch_save)
            self.assertIsNot(base.validate, original_validate)
        self.assertIs(base.deep_supervision_loss, original_loss)
        self.assertIs(base.validate, original_validate)
        self.assertIs(torch.save, original_save)


if __name__ == "__main__":
    unittest.main()
