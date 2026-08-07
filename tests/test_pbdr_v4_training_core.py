from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from experiments.pbdr_v4_state_contract import configure_stage_training
from experiments.pbdr_v4_training_core import (
    PBDRV4TrainingCoreError,
    build_candidate_checkpoint,
    build_optimizer,
    checkpoint_epoch_key,
    training_recipe,
    validate_candidate_checkpoint,
)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Conv2d(1, 2, 1)
        self.up_decoder1 = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.outc = nn.Conv2d(2, 1, 1)
        self.pbdr_v4 = nn.Conv2d(1, 1, 1)


def _metrics(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "intersection_pixels": 80,
        "union_pixels": 100,
        "matched_target_count": 9,
        "target_count": 10,
        "unmatched_component_pixels": 2,
        "valid_pixel_count": 1000,
        "matched_tiny_target_count": 2,
        "tiny_target_count": 3,
        "niou": 0.75,
        "test_loss": 0.1,
    }
    value.update(updates)
    return value


class PBDRV4TrainingCoreTests(unittest.TestCase):
    def test_epoch_key_has_no_pass_gate_and_ties_prefer_earlier_epoch(self) -> None:
        first = checkpoint_epoch_key("best_miou", _metrics(), 5)
        later = checkpoint_epoch_key("best_miou", _metrics(), 10)
        self.assertGreater(first, later)
        improved = checkpoint_epoch_key("best_miou", _metrics(intersection_pixels=81), 10)
        self.assertGreater(improved, first)
        self.assertEqual(len(first), 7)

    def test_role_keys_use_exact_pd_fa_and_miou_counts(self) -> None:
        base = checkpoint_epoch_key("best_pd", _metrics(), 5)
        less_fa = checkpoint_epoch_key(
            "best_pd", _metrics(unmatched_component_pixels=1), 5
        )
        self.assertGreater(less_fa, base)

    def test_optimizer_groups_are_exact_for_both_stages(self) -> None:
        stage1 = _Tiny()
        configure_stage_training(stage1, "stage1")
        optimizer1 = build_optimizer(stage1, "stage1")
        self.assertEqual([group["name"] for group in optimizer1.param_groups], ["pbdr_v4"])
        stage2 = _Tiny()
        configure_stage_training(stage2, "stage2")
        optimizer2 = build_optimizer(stage2, "stage2")
        self.assertEqual(
            [group["name"] for group in optimizer2.param_groups],
            ["pbdr_v4", "outc", "up_decoder1"],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer2.param_groups],
            [1e-4, 2e-6, 1e-6],
        )

    def test_recipe_has_no_performance_gate(self) -> None:
        self.assertIsNone(training_recipe("stage1")["performance_acceptance_margin"])
        self.assertEqual(training_recipe("stage1")["epochs"], 150)
        self.assertEqual(training_recipe("stage2")["epochs"], 50)

    def test_complete_candidate_payload_replays_and_rejects_wrong_role(self) -> None:
        model = _Tiny()
        metrics = _metrics()
        key = checkpoint_epoch_key("best_miou", metrics, 5)
        payload = build_candidate_checkpoint(
            dataset="NUDT-SIRST",
            role="best_miou",
            stage="stage1",
            epoch=5,
            architecture_manifest={"model": "tiny"},
            state_dict=model.state_dict(),
            validation_metrics=metrics,
            selection_key=key,
            parent_checkpoint_sha256="a" * 64,
            parent_state_sha256="b" * 64,
            split_projection_sha256="c" * 64,
            atlas_manifest_sha256="d" * 64,
            source_lock_sha256="e" * 64,
            initialization_checkpoint_sha256=None,
        )
        validate_candidate_checkpoint(
            payload,
            dataset="NUDT-SIRST",
            role="best_miou",
            stage="stage1",
        )
        with self.assertRaisesRegex(PBDRV4TrainingCoreError, "dataset/role/stage"):
            validate_candidate_checkpoint(
                payload,
                dataset="NUDT-SIRST",
                role="best_pd",
                stage="stage1",
            )


if __name__ == "__main__":
    unittest.main()
