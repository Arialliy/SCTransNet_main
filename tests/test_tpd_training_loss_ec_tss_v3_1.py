"""CPU contract tests for the frozen EC-TSS V3.1 objective."""

from __future__ import annotations

import inspect
import math
import unittest

import torch
import torch.nn as nn

from experiments.tpd_training_loss import compute_tpd_training_loss
from experiments.tpd_training_loss_ec_tss_v3_1 import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_SURVIVAL_RATIO_CAP,
    DEFAULT_SURVIVAL_WEIGHT,
    DEFAULT_TARGET_DILATION_RADIUS,
    ECTSSV31LossError,
    build_ec_tss_v3_1_risk_maps,
    compute_ec_tss_v3_1_training_loss,
    compute_error_conditioned_endpoint_terms,
)
from model.tpd_forward_contract import TPDForwardOutput


class ECTSSV31LossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = nn.BCELoss(reduction="mean")

    @staticmethod
    def structured(
        maps: tuple[torch.Tensor, ...],
        logit1: torch.Tensor | None = None,
        logit2: torch.Tensor | None = None,
    ) -> TPDForwardOutput:
        height, width = maps[-1].shape[-2:]
        shape = (maps[-1].shape[0], 1, height // 16, width // 16)
        if logit1 is None:
            logit1 = torch.zeros(shape, dtype=maps[-1].dtype)
        if logit2 is None:
            logit2 = torch.zeros(shape, dtype=maps[-1].dtype)
        return TPDForwardOutput(
            segmentation=maps,  # type: ignore[arg-type]
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )

    @staticmethod
    def constant_maps(value: float, *, size: int = 32) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.full((1, 1, size, size), value, dtype=torch.float32)
            for _ in range(6)
        )

    def test_frozen_defaults_and_no_survival_pos_weight_parameter(self) -> None:
        self.assertEqual(DEFAULT_SURVIVAL_WEIGHT, 0.005)
        self.assertEqual(DEFAULT_SURVIVAL_RATIO_CAP, 0.10)
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.5)
        self.assertEqual(DEFAULT_TARGET_DILATION_RADIUS, 3)
        signature = inspect.signature(compute_ec_tss_v3_1_training_loss)
        self.assertNotIn("survival_pos_weight", signature.parameters)
        with self.assertRaises(TypeError):
            compute_ec_tss_v3_1_training_loss(  # type: ignore[call-arg]
                self.structured(self.constant_maps(0.25)),
                torch.zeros(1, 1, 32, 32),
                self.criterion,
                survival_pos_weight=100.0,
            )

    def test_segmentation_loss_is_exact_historical_zero_survival_path(self) -> None:
        generator = torch.Generator().manual_seed(7)
        maps = tuple(
            torch.rand((1, 1, 32, 32), generator=generator).clamp(0.01, 0.99)
            for _ in range(6)
        )
        target = (torch.rand((1, 1, 32, 32), generator=generator) > 0.9).float()
        output = self.structured(maps)
        historical = compute_tpd_training_loss(
            output,
            target,
            self.criterion,
            survival_weight=0.0,
        )
        ec = compute_ec_tss_v3_1_training_loss(output, target, self.criterion)
        self.assertTrue(torch.equal(ec.segmentation, historical.segmentation))
        self.assertEqual(len(ec.segmentation_terms), 6)
        for observed, expected in zip(
            ec.segmentation_terms, historical.segmentation_terms
        ):
            self.assertTrue(torch.equal(observed, expected))

    def test_radius_three_dilation_and_separate_target_background_pooling(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        final = torch.zeros_like(target)
        # Same stride-16 cell as the target but outside its radius-3 neighborhood.
        final[0, 0, 15, 15] = 1.0
        # A true hard background response in a pure-background cell.
        final[0, 0, 20, 20] = 1.0
        risks = build_ec_tss_v3_1_risk_maps(final, target)

        self.assertEqual(float(risks.target_neighborhood[0, 0, 5, 5]), 1.0)
        self.assertEqual(float(risks.target_neighborhood[0, 0, 11, 11]), 1.0)
        self.assertEqual(float(risks.target_neighborhood[0, 0, 4, 8]), 0.0)
        self.assertEqual(float(risks.target_probability16[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(risks.background_probability16[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(risks.positive_risk[0, 0, 0, 0]), 1.0)
        # A target-containing cell is never treated as a negative-presence cell.
        self.assertEqual(float(risks.negative_risk[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(risks.negative_risk[0, 0, 1, 1]), 1.0)
        for value in (
            risks.target16,
            risks.target_neighborhood,
            risks.target_probability16,
            risks.background_probability16,
            risks.positive_risk,
            risks.negative_risk,
        ):
            self.assertFalse(value.requires_grad)

    def test_final_evaluator_prediction_not_other_deep_supervision_maps_drives_risk(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        auxiliary = torch.zeros_like(target)
        auxiliary[0, 0, 8, 8] = 1.0
        final = torch.zeros_like(target)
        output = self.structured((*tuple(auxiliary.clone() for _ in range(5)), final))
        losses = compute_ec_tss_v3_1_training_loss(
            output,
            target,
            self.criterion,
            survival_ratio_cap=None,
        )
        self.assertEqual(float(losses.positive_risk_mass), 1.0)
        self.assertEqual(int(losses.positive_active_cells), 1)

    def test_threshold_half_clamps_positive_and_negative_risks(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        final = torch.zeros_like(target)
        final[0, 0, 8, 8] = 0.25
        final[0, 0, 20, 20] = 0.75
        risks = build_ec_tss_v3_1_risk_maps(final, target)
        self.assertAlmostEqual(float(risks.positive_risk[0, 0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(risks.negative_risk[0, 0, 1, 1]), 0.5)

    def test_risk_mass_normalization_is_not_diluted_by_zero_risk_cells(self) -> None:
        one_logit = torch.zeros(1, 1, 1, 1)
        one_positive = torch.zeros_like(one_logit)
        one_negative = torch.ones_like(one_logit)
        _, reference = compute_error_conditioned_endpoint_terms(
            one_logit, one_positive, one_negative
        )

        many_logits = torch.zeros(1, 1, 1, 101)
        many_positive = torch.zeros_like(many_logits)
        many_negative = torch.zeros_like(many_logits)
        many_negative[..., 0] = 1.0
        # The 100 added cells may have arbitrary logits: their risk is exactly zero.
        many_logits[..., 1:] = 50.0
        _, observed = compute_error_conditioned_endpoint_terms(
            many_logits, many_positive, many_negative
        )
        self.assertTrue(torch.equal(observed, reference))

    def test_no_target_crop_has_zero_positive_and_finite_negative_branch(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        maps = list(self.constant_maps(0.01))
        maps[-1] = maps[-1].clone()
        maps[-1][0, 0, 20, 20] = 0.9
        losses = compute_ec_tss_v3_1_training_loss(
            self.structured(tuple(maps)), target, self.criterion
        )
        self.assertEqual(float(losses.positive_survival), 0.0)
        self.assertEqual(float(losses.positive_risk_mass), 0.0)
        self.assertEqual(int(losses.positive_active_cells), 0)
        self.assertGreater(float(losses.negative_survival), 0.0)
        self.assertGreater(float(losses.negative_risk_mass), 0.0)
        self.assertEqual(int(losses.negative_active_cells), 1)
        self.assertTrue(math.isfinite(float(losses.total)))

    def test_perfect_threshold_state_strictly_exits_both_branches(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        maps = tuple(target.clone() for _ in range(6))
        losses = compute_ec_tss_v3_1_training_loss(
            self.structured(maps), target, self.criterion
        )
        for value in (
            losses.survival,
            losses.positive_survival,
            losses.negative_survival,
            losses.weighted_survival,
            losses.positive_risk_mass,
            losses.negative_risk_mass,
        ):
            self.assertEqual(float(value), 0.0)
        self.assertTrue(torch.equal(losses.total, losses.segmentation))

    def test_stop_gradient_isolates_final_probability_but_updates_logits(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        final = torch.zeros_like(target, requires_grad=True)
        maps = (*self.constant_maps(0.1)[:5], final)
        logit1 = torch.zeros(1, 1, 2, 2, requires_grad=True)
        logit2 = torch.zeros(1, 1, 2, 2, requires_grad=True)
        losses = compute_ec_tss_v3_1_training_loss(
            self.structured(maps, logit1, logit2),
            target,
            self.criterion,
            survival_ratio_cap=None,
        )
        final_gradient = torch.autograd.grad(
            losses.survival,
            final,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNone(final_gradient)
        logit_gradients = torch.autograd.grad(
            losses.survival,
            (logit1, logit2),
        )
        self.assertTrue(all(gradient is not None for gradient in logit_gradients))
        self.assertTrue(all(torch.count_nonzero(gradient) > 0 for gradient in logit_gradients))

    def test_four_terms_use_exact_quarter_aggregation_and_branch_diagnostics(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        target[0, 0, 8, 8] = 1.0
        final = torch.zeros_like(target)
        final[0, 0, 20, 20] = 1.0
        maps = (*self.constant_maps(0.1)[:5], final)
        logit1 = torch.tensor([[[[-1.0, 0.0], [0.0, 2.0]]]])
        logit2 = torch.tensor([[[[3.0, 0.0], [0.0, -4.0]]]])
        losses = compute_ec_tss_v3_1_training_loss(
            self.structured(maps, logit1, logit2),
            target,
            self.criterion,
            survival_ratio_cap=None,
        )
        four_terms = (
            losses.endpoint_positive_terms[0]
            + losses.endpoint_negative_terms[0]
            + losses.endpoint_positive_terms[1]
            + losses.endpoint_negative_terms[1]
        )
        self.assertTrue(torch.equal(losses.survival, 0.25 * four_terms))
        self.assertTrue(
            torch.equal(
                losses.positive_survival,
                0.5
                * (
                    losses.endpoint_positive_terms[0]
                    + losses.endpoint_positive_terms[1]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                losses.negative_survival,
                0.5
                * (
                    losses.endpoint_negative_terms[0]
                    + losses.endpoint_negative_terms[1]
                ),
            )
        )

    def test_cap_inactive_and_active_contract(self) -> None:
        target = torch.zeros(1, 1, 32, 32)

        inactive_maps = list(self.constant_maps(0.01))
        inactive_maps[-1] = inactive_maps[-1].clone()
        inactive_maps[-1][0, 0, 20, 20] = 0.6
        inactive = compute_ec_tss_v3_1_training_loss(
            self.structured(tuple(inactive_maps)), target, self.criterion
        )
        self.assertAlmostEqual(
            float(inactive.effective_survival_weight), DEFAULT_SURVIVAL_WEIGHT
        )
        self.assertFalse(inactive.effective_survival_weight.requires_grad)

        active_maps = list(self.constant_maps(0.001))
        active_maps[-1] = active_maps[-1].clone()
        active_maps[-1][0, 0, 20, 20] = 0.9
        large_logit = torch.full((1, 1, 2, 2), 100.0)
        active = compute_ec_tss_v3_1_training_loss(
            self.structured(tuple(active_maps), large_logit, large_logit.clone()),
            target,
            self.criterion,
        )
        self.assertLess(
            float(active.effective_survival_weight), DEFAULT_SURVIVAL_WEIGHT
        )
        self.assertLessEqual(
            float(active.weighted_survival),
            DEFAULT_SURVIVAL_RATIO_CAP * float(active.segmentation) + 1e-7,
        )
        self.assertFalse(active.effective_survival_weight.requires_grad)

    def test_invalid_contracts_fail_closed(self) -> None:
        target = torch.zeros(1, 1, 32, 32)
        output = self.structured(self.constant_maps(0.1))
        for threshold in (0.0, 1.0, -0.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaises(
                ECTSSV31LossError
            ):
                compute_ec_tss_v3_1_training_loss(
                    output,
                    target,
                    self.criterion,
                    confidence_threshold=threshold,
                )
        for radius in (-1, 1.5, True):
            with self.subTest(radius=radius), self.assertRaises(ECTSSV31LossError):
                compute_ec_tss_v3_1_training_loss(
                    output,
                    target,
                    self.criterion,
                    target_dilation_radius=radius,  # type: ignore[arg-type]
                )
        with self.assertRaisesRegex(ECTSSV31LossError, "structured"):
            compute_ec_tss_v3_1_training_loss(
                self.constant_maps(0.1),  # type: ignore[arg-type]
                target,
                self.criterion,
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            small_target = torch.zeros(1, 1, 30, 32)
            small_maps = tuple(torch.full_like(small_target, 0.1) for _ in range(6))
            compute_ec_tss_v3_1_training_loss(
                self.structured(small_maps), small_target, self.criterion
            )


if __name__ == "__main__":
    unittest.main()
