from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from model.tpd_forward_contract import evaluator_prediction, legacy_output
from model.tpd_survival import (
    PairedTargetSurvivalHeads,
    SURVIVAL_ENDPOINT_CONTRACT,
    TargetSurvivalHead,
    build_structured_survival_output,
    survival_parameter_count,
)


class TargetSurvivalHeadTests(unittest.TestCase):
    def test_head_returns_raw_single_channel_logits(self) -> None:
        head = TargetSurvivalHead(4)
        endpoint = torch.randn(2, 4, 3, 5)
        logits = head(endpoint)
        self.assertEqual(tuple(logits.shape), (2, 1, 3, 5))
        self.assertFalse(
            torch.equal(logits, torch.sigmoid(logits)),
            "head must return logits rather than probabilities",
        )

    def test_head_validates_constructor_and_endpoint_contract(self) -> None:
        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(in_channels=value):
                with self.assertRaises(ValueError):
                    TargetSurvivalHead(value)  # type: ignore[arg-type]

        head = TargetSurvivalHead(4)
        invalid = (
            torch.randn(2, 4, 5),
            torch.randn(2, 3, 4, 5),
            torch.ones(2, 4, 3, 5, dtype=torch.int64),
            torch.full((2, 4, 3, 5), float("nan")),
        )
        for endpoint in invalid:
            with self.subTest(shape=tuple(endpoint.shape), dtype=endpoint.dtype):
                with self.assertRaises(
                    (ValueError, TypeError, FloatingPointError)
                ):
                    head(endpoint)


class PairedTargetSurvivalHeadsTests(unittest.TestCase):
    def test_pair_uses_exact_emb1_emb2_grid_and_parameter_count(self) -> None:
        heads = PairedTargetSurvivalHeads(4, 8)
        emb1 = torch.randn(2, 4, 3, 5)
        emb2 = torch.randn(2, 8, 3, 5)
        logits1, logits2 = heads(emb1, emb2)
        self.assertEqual(tuple(logits1.shape), (2, 1, 3, 5))
        self.assertEqual(tuple(logits2.shape), (2, 1, 3, 5))
        self.assertEqual(survival_parameter_count(heads), (4 + 1) + (8 + 1))
        self.assertEqual(
            set(dict(heads.named_parameters())),
            {
                "heads.emb1.classifier.weight",
                "heads.emb1.classifier.bias",
                "heads.emb2.classifier.weight",
                "heads.emb2.classifier.bias",
            },
        )

    def test_pair_rejects_channel_ratio_batch_and_grid_mismatch(self) -> None:
        for first, second in ((0, 8), (4, 0), (True, 2), (4, "8")):
            with self.subTest(channels=(first, second)):
                with self.assertRaises(ValueError):
                    PairedTargetSurvivalHeads(first, second)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "twice"):
            PairedTargetSurvivalHeads(4, 9)

        heads = PairedTargetSurvivalHeads(4, 8)
        valid1 = torch.randn(2, 4, 3, 5)
        valid2 = torch.randn(2, 8, 3, 5)
        cases = (
            (valid1[:1], valid2, "batch"),
            (valid1, torch.randn(2, 8, 2, 5), "spatial"),
            (torch.randn(2, 3, 3, 5), valid2, "channels"),
            (valid1, torch.randn(2, 7, 3, 5), "channels"),
        )
        for first, second, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    heads(first, second)

    def test_manifest_keeps_auxiliary_head_out_of_segmentation_path(self) -> None:
        manifest = PairedTargetSurvivalHeads().architecture_manifest()
        self.assertEqual(manifest["supervised_endpoints"], ("emb1", "emb2"))
        self.assertEqual(manifest["target_grid"], "stride_16_max_presence")
        self.assertEqual(
            manifest["endpoint_contract"],
            SURVIVAL_ENDPOINT_CONTRACT,
        )
        self.assertFalse(manifest["segmentation_path_modified"])
        self.assertFalse(manifest["inference_heads_required"])

    def test_structured_output_preserves_legacy_evaluator_prediction(self) -> None:
        heads = PairedTargetSurvivalHeads(4, 8)
        segmentation = tuple(
            torch.sigmoid(torch.randn(2, 1, 48, 80)) for _ in range(6)
        )
        emb1 = torch.randn(2, 4, 3, 5)
        emb2 = torch.randn(2, 8, 3, 5)
        structured = build_structured_survival_output(
            segmentation,
            emb1,
            emb2,
            heads,
        )
        self.assertIs(legacy_output(structured), segmentation)
        self.assertIs(evaluator_prediction(structured), segmentation[-1])
        self.assertEqual(structured.token_endpoints, (emb1, emb2))
        self.assertEqual(
            tuple(tuple(value.shape) for value in structured.survival_logits),
            ((2, 1, 3, 5), (2, 1, 3, 5)),
        )

    def test_survival_loss_backpropagates_to_heads_and_endpoints(self) -> None:
        torch.manual_seed(19)
        heads = PairedTargetSurvivalHeads(4, 8)
        emb1 = torch.randn(2, 4, 2, 3, requires_grad=True)
        emb2 = torch.randn(2, 8, 2, 3, requires_grad=True)
        logits = heads(emb1, emb2)
        target = torch.randint(0, 2, (2, 1, 2, 3)).float()
        loss = sum(
            nn.functional.binary_cross_entropy_with_logits(value, target)
            for value in logits
        )
        loss.backward()
        self.assertGreater(float(emb1.grad.abs().sum()), 0.0)
        self.assertGreater(float(emb2.grad.abs().sum()), 0.0)
        for name, parameter in heads.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
