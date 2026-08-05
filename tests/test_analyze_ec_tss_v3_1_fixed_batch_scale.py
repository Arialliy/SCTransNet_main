"""CPU tests for the EC-TSS V3.1 fixed-first-train-batch audit."""

from __future__ import annotations

import json
import math
import unittest

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from analysis.analyze_ec_tss_v3_1_fixed_batch_scale import (
    AUDIT_BATCH_SIZE,
    AUDIT_EPOCH,
    ECTSSV31ScaleAuditError,
    audit_losses_and_shared_gradients,
    build_fixed_first_train_batch,
    parse_args,
    shared_named_parameters,
)
from experiments import train_three_dataset_ec_tss_v3_1_seed42 as runner
from model.tpd_forward_contract import TPDForwardOutput


class _TinySurvivalHeads(nn.Module):
    def __init__(self, head_value: float) -> None:
        super().__init__()
        self.emb1 = nn.Parameter(torch.tensor(head_value, dtype=torch.float32))
        self.emb2 = nn.Parameter(torch.tensor(head_value, dtype=torch.float32))


class _TinyStructuredModel(nn.Module):
    """One shared scalar plus two parameters under the formal head prefix."""

    def __init__(self, head_value: float) -> None:
        super().__init__()
        self.shared_logit = nn.Parameter(torch.tensor(-0.4, dtype=torch.float32))
        self.target_survival = _TinySurvivalHeads(head_value)

    def forward(self, images: torch.Tensor) -> TPDForwardOutput:
        batch, _, height, width = images.shape
        probability = torch.sigmoid(self.shared_logit).expand(
            batch, 1, height, width
        )
        token = self.shared_logit.expand(batch, 1, height // 16, width // 16)
        return TPDForwardOutput(
            segmentation=tuple(probability for _ in range(6)),  # type: ignore[arg-type]
            emb1_survival_logits=token * self.target_survival.emb1,
            emb2_survival_logits=token * self.target_survival.emb2,
        )


class _MetadataDataset(Dataset[dict[str, object]]):
    def __init__(self) -> None:
        self.epoch: int | None = None

    def __len__(self) -> int:
        return 20

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        if self.epoch is None:
            raise RuntimeError("set_epoch must precede iteration")
        image = torch.full((1, 32, 32), float(index) / 20.0)
        mask = torch.zeros(1, 32, 32)
        mask[0, index % 32, (index * 3) % 32] = 1.0
        return {
            "image": image,
            "mask": mask,
            "namespaced_sample_id": f"sample:{index:03d}",
            # Exercise the real protocol's unsigned seeds above int64 range.
            "augmentation_seed": (1 << 63) + self.epoch * 10_000 + index,
        }


class ECTSSV31FixedBatchScaleAuditTests(unittest.TestCase):
    @staticmethod
    def _batch() -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.zeros(2, 1, 32, 32)
        masks = torch.zeros_like(images)
        masks[:, :, 8, 8] = 1.0
        return images, masks

    def test_shared_selection_excludes_only_target_survival_prefix(self) -> None:
        model = _TinyStructuredModel(head_value=1.0)
        selected = shared_named_parameters(model)
        self.assertEqual([name for name, _ in selected], ["shared_logit"])
        self.assertEqual(sum(parameter.numel() for _, parameter in selected), 1)

    def test_audit_reports_losses_risks_cap_and_shared_gradient_ratio(self) -> None:
        model = _TinyStructuredModel(head_value=1.0)
        images, masks = self._batch()
        observed = audit_losses_and_shared_gradients(model, images, masks)

        self.assertGreater(observed["losses"]["segmentation"], 0.0)
        self.assertGreater(observed["losses"]["ec_tss"], 0.0)
        self.assertGreater(observed["risk"]["positive_risk_mass"], 0.0)
        self.assertEqual(observed["risk"]["negative_risk_mass"], 0.0)
        self.assertIsInstance(observed["losses"]["cap_active"], bool)
        self.assertEqual(
            observed["losses"]["weighted_ec_to_segmentation_ratio"],
            observed["losses"]["weighted_ec_tss"]
            / observed["losses"]["segmentation"],
        )
        gradients = observed["shared_parameter_gradients"]
        self.assertTrue(gradients["survival_head_parameters_excluded"])
        self.assertEqual(gradients["selected_parameter_count"], 1)
        self.assertGreater(gradients["segmentation"]["global_l2_norm"], 0.0)
        self.assertGreater(gradients["weighted_ec_tss"]["global_l2_norm"], 0.0)
        self.assertGreater(
            gradients["weighted_ec_to_segmentation_global_l2_ratio"], 0.0
        )
        self.assertTrue(
            math.isfinite(
                gradients["weighted_ec_to_segmentation_global_l2_ratio"]
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.parameters())
        )
        json.dumps(observed, allow_nan=False)

    def test_zero_initialized_heads_can_have_zero_shared_ec_gradient(self) -> None:
        model = _TinyStructuredModel(head_value=0.0)
        images, masks = self._batch()
        observed = audit_losses_and_shared_gradients(model, images, masks)
        gradients = observed["shared_parameter_gradients"]
        self.assertGreater(gradients["segmentation"]["global_l2_norm"], 0.0)
        self.assertEqual(gradients["weighted_ec_tss"]["global_l2_norm"], 0.0)
        self.assertEqual(
            gradients["weighted_ec_to_segmentation_global_l2_ratio"], 0.0
        )

    def test_fixed_epoch_one_batch_is_reproducible_and_metadata_complete(self) -> None:
        first_dataset = _MetadataDataset()
        first_images, first_masks, first = build_fixed_first_train_batch(
            first_dataset,
            runner.DATASETS[0],
            device=torch.device("cpu"),
        )
        torch.manual_seed(999_999)
        second_dataset = _MetadataDataset()
        second_images, second_masks, second = build_fixed_first_train_batch(
            second_dataset,
            runner.DATASETS[0],
            device=torch.device("cpu"),
        )

        self.assertEqual(first_dataset.epoch, AUDIT_EPOCH)
        self.assertEqual(second_dataset.epoch, AUDIT_EPOCH)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(first_images, second_images))
        self.assertTrue(torch.equal(first_masks, second_masks))
        self.assertEqual(first["batch_size"], AUDIT_BATCH_SIZE)
        self.assertEqual(len(first["namespaced_sample_ids"]), AUDIT_BATCH_SIZE)
        self.assertEqual(len(first["augmentation_seeds"]), AUDIT_BATCH_SIZE)
        self.assertTrue(
            all(value > (1 << 63) for value in first["augmentation_seeds"])
        )
        self.assertEqual(len(first["images_sha256"]), 64)
        self.assertEqual(len(first["masks_sha256"]), 64)

    def test_batch_size_and_cli_dataset_are_frozen(self) -> None:
        with self.assertRaises(ECTSSV31ScaleAuditError):
            build_fixed_first_train_batch(
                _MetadataDataset(),
                runner.DATASETS[0],
                batch_size=AUDIT_BATCH_SIZE - 1,
                device=torch.device("cpu"),
            )
        args = parse_args(["--dataset", runner.DATASETS[0], "--device", "cpu"])
        self.assertEqual(args.dataset, runner.DATASETS[0])
        self.assertEqual(args.device, "cpu")
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "SIRST3", "--device", "cpu"])


if __name__ == "__main__":
    unittest.main()
