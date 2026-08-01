from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.tpd_training_loss import (
    TPDTrainingLossError,
    build_survival_target,
    compute_tpd_training_loss,
)
from model.tpd_forward_contract import TPDForwardOutput


class TPDTrainingLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260726)
        self.target = (torch.rand(2, 1, 32, 48) > 0.9).float()
        self.criterion = nn.BCELoss(reduction="mean")

    def test_single_and_six_map_segmentation_match_legacy_objective(self) -> None:
        single = torch.sigmoid(torch.randn_like(self.target))
        single_result = compute_tpd_training_loss(
            single,
            self.target,
            self.criterion,
        )
        torch.testing.assert_close(
            single_result.total,
            self.criterion(single, self.target),
        )
        self.assertEqual(len(single_result.segmentation_terms), 1)
        self.assertEqual(single_result.survival_terms, ())
        self.assertEqual(single_result.survival.item(), 0.0)

        six = tuple(
            torch.sigmoid(torch.randn_like(self.target)) for _ in range(6)
        )
        six_result = compute_tpd_training_loss(
            six,
            self.target,
            self.criterion,
        )
        expected = sum(self.criterion(item, self.target) for item in six)
        self.assertTrue(torch.equal(six_result.total, expected))
        self.assertEqual(len(six_result.segmentation_terms), 6)

    def test_zero_weight_is_exact_no_auxiliary_path(self) -> None:
        segmentation = tuple(
            torch.sigmoid(torch.randn_like(self.target)).requires_grad_()
            for _ in range(6)
        )
        logit1 = torch.full(
            (2, 1, 2, 3),
            float("nan"),
            requires_grad=True,
        )
        logit2 = torch.full(
            (2, 1, 2, 3),
            float("nan"),
            requires_grad=True,
        )
        result = TPDForwardOutput(
            segmentation=segmentation,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )
        loss = compute_tpd_training_loss(
            result,
            self.target,
            self.criterion,
            survival_weight=0.0,
        )
        expected = sum(self.criterion(item, self.target) for item in segmentation)
        self.assertTrue(torch.equal(loss.total, expected))
        self.assertEqual(loss.survival_terms, ())
        loss.total.backward()
        self.assertIsNone(logit1.grad)
        self.assertIsNone(logit2.grad)

    def test_survival_target_is_exact_fixed_max_pool(self) -> None:
        target = torch.zeros(1, 1, 32, 48)
        target[0, 0, 15, 16] = 1
        target[0, 0, 31, 47] = 1
        pooled = build_survival_target(target)
        expected = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]])
        torch.testing.assert_close(pooled, expected)
        torch.testing.assert_close(
            build_survival_target(torch.zeros_like(target)),
            torch.zeros_like(expected),
        )

    def test_positive_weight_uses_two_logits_and_formula(self) -> None:
        segmentation = torch.sigmoid(torch.randn_like(self.target)).requires_grad_()
        logit1 = torch.randn(2, 1, 2, 3, requires_grad=True)
        logit2 = torch.randn(2, 1, 2, 3, requires_grad=True)
        result = TPDForwardOutput(
            segmentation=segmentation,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )
        weight = 0.01
        pos_weight = 10.116
        losses = compute_tpd_training_loss(
            result,
            self.target,
            self.criterion,
            survival_weight=weight,
            survival_pos_weight=pos_weight,
        )

        pooled = F.max_pool2d(self.target, 16, 16)
        pw = torch.tensor(pos_weight).reshape(1, 1, 1)
        expected_survival = sum(
            F.binary_cross_entropy_with_logits(logit, pooled, pos_weight=pw)
            for logit in (logit1, logit2)
        )
        expected_segmentation = self.criterion(segmentation, self.target)
        torch.testing.assert_close(losses.survival, expected_survival)
        torch.testing.assert_close(losses.segmentation, expected_segmentation)
        torch.testing.assert_close(
            losses.total,
            expected_segmentation + weight * expected_survival,
        )
        torch.testing.assert_close(
            losses.effective_survival_weight,
            torch.tensor(weight),
        )
        torch.testing.assert_close(
            losses.weighted_survival,
            weight * expected_survival,
        )

        losses.total.backward()
        self.assertGreater(float(segmentation.grad.abs().sum()), 0.0)
        self.assertGreater(float(logit1.grad.abs().sum()), 0.0)
        self.assertGreater(float(logit2.grad.abs().sum()), 0.0)

    def test_survival_ratio_cap_limits_auxiliary_contribution(self) -> None:
        segmentation = torch.full_like(self.target, 0.9, requires_grad=True)
        logit1 = torch.zeros(2, 1, 2, 3, requires_grad=True)
        logit2 = torch.zeros(2, 1, 2, 3, requires_grad=True)
        output = TPDForwardOutput(
            segmentation=segmentation,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )
        requested_weight = 0.5
        ratio_cap = 0.1
        losses = compute_tpd_training_loss(
            output,
            self.target,
            self.criterion,
            survival_weight=requested_weight,
            survival_pos_weight=10.116,
            survival_ratio_cap=ratio_cap,
        )
        expected_weight = torch.minimum(
            losses.segmentation.new_tensor(requested_weight),
            ratio_cap
            * losses.segmentation.detach()
            / losses.survival.detach().clamp_min(
                torch.finfo(losses.survival.dtype).eps
            ),
        )
        torch.testing.assert_close(losses.effective_survival_weight, expected_weight)
        torch.testing.assert_close(
            losses.weighted_survival,
            expected_weight * losses.survival,
        )
        torch.testing.assert_close(
            losses.total,
            losses.segmentation + losses.weighted_survival,
        )
        self.assertLessEqual(
            float(losses.weighted_survival.detach()),
            ratio_cap * float(losses.segmentation.detach()) + 1e-6,
        )

        losses.total.backward()
        self.assertIsNone(losses.effective_survival_weight.grad_fn)
        self.assertGreater(float(segmentation.grad.abs().sum()), 0.0)
        self.assertGreater(float(logit1.grad.abs().sum()), 0.0)
        self.assertGreater(float(logit2.grad.abs().sum()), 0.0)

    def test_auxiliary_logits_never_enter_segmentation_terms(self) -> None:
        segmentation = tuple(
            torch.sigmoid(torch.randn_like(self.target)) for _ in range(6)
        )
        result = TPDForwardOutput(
            segmentation=segmentation,
            emb1_survival_logits=torch.randn(2, 1, 2, 3),
            emb2_survival_logits=torch.randn(2, 1, 2, 3),
        )
        losses = compute_tpd_training_loss(
            result,
            self.target,
            self.criterion,
            survival_weight=0.01,
        )
        self.assertEqual(len(losses.segmentation_terms), 6)
        self.assertEqual(len(losses.survival_terms), 2)

    def test_rejects_missing_logits_and_spatial_mismatch(self) -> None:
        prediction = torch.sigmoid(torch.randn_like(self.target))
        with self.assertRaisesRegex(
            TPDTrainingLossError,
            "structured TPDForwardOutput",
        ):
            compute_tpd_training_loss(
                prediction,
                self.target,
                self.criterion,
                survival_weight=0.01,
            )

        structured = TPDForwardOutput(segmentation=prediction)
        with self.assertRaisesRegex(
            TPDTrainingLossError,
            "both survival logits",
        ):
            compute_tpd_training_loss(
                structured,
                self.target,
                self.criterion,
                survival_weight=0.01,
            )

        mismatched = TPDForwardOutput(
            segmentation=prediction,
            emb1_survival_logits=torch.randn(2, 1, 1, 3),
            emb2_survival_logits=torch.randn(2, 1, 1, 3),
        )
        with self.assertRaisesRegex(TPDTrainingLossError, "does not match"):
            compute_tpd_training_loss(
                mismatched,
                self.target,
                self.criterion,
                survival_weight=0.01,
            )

    def test_rejects_invalid_target_weight_and_non_scalar_criterion(self) -> None:
        prediction = torch.sigmoid(torch.randn_like(self.target))
        with self.assertRaisesRegex(TPDTrainingLossError, "divisible"):
            build_survival_target(torch.zeros(1, 1, 31, 48))
        with self.assertRaisesRegex(TPDTrainingLossError, "non-negative"):
            compute_tpd_training_loss(
                prediction,
                self.target,
                self.criterion,
                survival_weight=-1.0,
            )
        with self.assertRaisesRegex(TPDTrainingLossError, "scalar Tensor"):
            compute_tpd_training_loss(
                prediction,
                self.target,
                nn.BCELoss(reduction="none"),
            )
        for value in (0.0, -1.0, float("nan"), True):
            with self.subTest(survival_ratio_cap=value):
                with self.assertRaisesRegex(
                    TPDTrainingLossError,
                    "survival_ratio_cap",
                ):
                    compute_tpd_training_loss(
                        prediction,
                        self.target,
                        self.criterion,
                        survival_ratio_cap=value,
                    )

    def test_rejects_invalid_pos_weight_and_non_finite_logits(self) -> None:
        prediction = torch.sigmoid(torch.randn_like(self.target))
        for value in (0.0, -1.0, float("nan"), torch.ones(2)):
            with self.subTest(pos_weight=value):
                output = TPDForwardOutput(
                    segmentation=prediction,
                    emb1_survival_logits=torch.randn(2, 1, 2, 3),
                    emb2_survival_logits=torch.randn(2, 1, 2, 3),
                )
                with self.assertRaisesRegex(
                    TPDTrainingLossError,
                    "survival_pos_weight",
                ):
                    compute_tpd_training_loss(
                        output,
                        self.target,
                        self.criterion,
                        survival_weight=0.01,
                        survival_pos_weight=value,
                    )

        for value in (float("nan"), float("inf")):
            with self.subTest(logit=value):
                output = TPDForwardOutput(
                    segmentation=prediction,
                    emb1_survival_logits=torch.full((2, 1, 2, 3), value),
                    emb2_survival_logits=torch.zeros(2, 1, 2, 3),
                )
                with self.assertRaisesRegex(
                    FloatingPointError,
                    "non-finite",
                ):
                    compute_tpd_training_loss(
                        output,
                        self.target,
                        self.criterion,
                        survival_weight=0.01,
                    )

    def test_explicitly_rejects_survival_logit_device_mismatch(self) -> None:
        prediction = torch.sigmoid(torch.randn_like(self.target))
        output = TPDForwardOutput(
            segmentation=prediction,
            emb1_survival_logits=torch.empty(2, 1, 2, 3, device="meta"),
            emb2_survival_logits=torch.empty(2, 1, 2, 3, device="meta"),
        )
        with self.assertRaisesRegex(TPDTrainingLossError, "same device"):
            compute_tpd_training_loss(
                output,
                self.target,
                self.criterion,
                survival_weight=0.01,
            )

    def test_fp32_auxiliary_loss_and_gradients_under_cpu_autocast(self) -> None:
        prediction = torch.sigmoid(torch.randn_like(self.target)).to(
            torch.bfloat16
        )
        prediction.requires_grad_()
        logit1 = torch.randn(2, 1, 2, 3).to(torch.bfloat16).requires_grad_()
        logit2 = torch.randn(2, 1, 2, 3).to(torch.bfloat16).requires_grad_()
        output = TPDForwardOutput(
            segmentation=prediction,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = compute_tpd_training_loss(
                output,
                self.target,
                self.criterion,
                survival_weight=0.01,
            )
        self.assertEqual(result.total.dtype, torch.float32)
        self.assertEqual(result.segmentation.dtype, torch.float32)
        self.assertEqual(result.survival.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        for gradient in (prediction.grad, logit1.grad, logit2.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
