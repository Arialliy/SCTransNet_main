from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from experiments.pbdr_v4_state_contract import (
    PBDRV4StateContractError,
    audit_candidate_against_current,
    audit_training_modes,
    checkpoint_epoch_key,
    clone_current_state,
    configure_stage_training,
    l2sp_to_current,
)
from tests.test_pbdr_v4_zero_margin_selector import _record


class _TinyModel(nn.Module):
    def __init__(self, *, with_pbdr: bool) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(1, 2, 1)
        self.up_decoder1 = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False),
            nn.BatchNorm2d(2),
        )
        self.outc = nn.Conv2d(2, 1, 1)
        if with_pbdr:
            self.pbdr_v4 = nn.Conv2d(1, 1, 1)


class PBDRV4StateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.current_model = _TinyModel(with_pbdr=False)
        self.current = clone_current_state(self.current_model)
        self.candidate = _TinyModel(with_pbdr=True)
        base = self.candidate.state_dict()
        for name, value in self.current.items():
            base[name].copy_(value)
        self.candidate.load_state_dict(base)

    def test_stage1_trainable_set_and_base_identity(self) -> None:
        names = configure_stage_training(self.candidate, "stage1")
        self.assertEqual(names, ("pbdr_v4.bias", "pbdr_v4.weight"))
        report = audit_candidate_against_current(
            self.candidate,
            current_state=self.current,
            stage="stage1",
        )
        self.assertEqual(report["permitted_changed_parameter_names"], [])

    def test_stage2_allows_only_parameter_prefixes_and_keeps_bn_eval(self) -> None:
        names = configure_stage_training(self.candidate, "stage2")
        self.assertIn("outc.weight", names)
        self.assertIn("up_decoder1.0.weight", names)
        self.assertIn("up_decoder1.1.weight", names)
        self.assertFalse(self.candidate.training)
        self.assertTrue(self.candidate.pbdr_v4.training)
        self.assertFalse(self.candidate.up_decoder1[1].training)
        with torch.no_grad():
            self.candidate.outc.weight.add_(1.0)
            self.candidate.up_decoder1[1].weight.add_(1.0)
        audit_candidate_against_current(
            self.candidate,
            current_state=self.current,
            stage="stage2",
        )

    def test_stage2_bn_buffer_change_is_never_allowed(self) -> None:
        configure_stage_training(self.candidate, "stage2")
        self.candidate.up_decoder1[1].running_mean.add_(1.0)
        with self.assertRaisesRegex(PBDRV4StateContractError, "buffer"):
            audit_candidate_against_current(
                self.candidate,
                current_state=self.current,
                stage="stage2",
            )

    def test_stage1_base_or_stage2_other_change_is_rejected(self) -> None:
        configure_stage_training(self.candidate, "stage1")
        with torch.no_grad():
            self.candidate.outc.bias.add_(1.0)
        with self.assertRaisesRegex(PBDRV4StateContractError, "immutable parameter"):
            audit_candidate_against_current(
                self.candidate,
                current_state=self.current,
                stage="stage1",
            )
        self.setUp()
        configure_stage_training(self.candidate, "stage2")
        with torch.no_grad():
            self.candidate.backbone.weight.add_(1.0)
        with self.assertRaisesRegex(PBDRV4StateContractError, "immutable parameter"):
            audit_candidate_against_current(
                self.candidate,
                current_state=self.current,
                stage="stage2",
            )

    def test_model_train_cannot_silently_reenable_bn(self) -> None:
        configure_stage_training(self.candidate, "stage2")
        self.candidate.train()
        with self.assertRaisesRegex(PBDRV4StateContractError, "eval mode|BatchNorm"):
            audit_training_modes(self.candidate, "stage2")

    def test_l2sp_is_anchored_to_current_not_stage1(self) -> None:
        configure_stage_training(self.candidate, "stage2")
        self.assertEqual(
            float(l2sp_to_current(self.candidate, current_state=self.current).detach()),
            0.0,
        )
        with torch.no_grad():
            self.candidate.outc.weight.add_(0.5)
        self.assertGreater(
            float(l2sp_to_current(self.candidate, current_state=self.current).detach()),
            0.0,
        )

    def test_epoch_key_has_full_role_key_then_earlier_epoch(self) -> None:
        metric = _record("candidate", "V4-Stage1")
        key1 = checkpoint_epoch_key(role="best_miou", metrics=metric, epoch=1)
        key2 = checkpoint_epoch_key(role="best_miou", metrics=metric, epoch=2)
        self.assertGreater(key1, key2)
        self.assertEqual(len(key1), 7)


if __name__ == "__main__":
    unittest.main()
