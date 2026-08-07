from __future__ import annotations

import unittest

import torch

from experiments.pbdr_v3_residual_calibration import (
    CURRENT_ANCHOR,
    PBDR_V3_ANCHOR,
    ResidualCalibration,
    apply_residual_calibration,
    calibration_grid,
)


class PBDRV3ResidualCalibrationTests(unittest.TestCase):
    def test_grid_is_pre_registered_unique_and_contains_both_anchors(self) -> None:
        grid = calibration_grid()
        self.assertEqual(len(grid), 7 * 6 * 9)
        self.assertEqual(len(set(grid)), len(grid))
        self.assertIn(CURRENT_ANCHOR, grid)
        self.assertIn(PBDR_V3_ANCHOR, grid)

    def test_current_and_v3_anchors_are_exact(self) -> None:
        base = torch.tensor([[[[-2.0, -0.1], [0.0, 3.0]]]])
        delta = torch.tensor([[[[-0.3, 0.2], [-0.4, 0.5]]]])
        self.assertIs(apply_residual_calibration(base, delta, CURRENT_ANCHOR), base)
        self.assertTrue(
            torch.equal(
                apply_residual_calibration(base, delta, PBDR_V3_ANCHOR),
                base + delta,
            )
        )

    def test_positive_and_negative_scales_are_independent(self) -> None:
        base = torch.zeros(1, 1, 1, 4)
        delta = torch.tensor([[[[-2.0, -1.0, 1.0, 2.0]]]])
        config = ResidualCalibration(positive_scale=3.0, negative_scale=0.5, bias=0.1)
        actual = apply_residual_calibration(base, delta, config)
        expected = torch.tensor([[[[-0.9, -0.4, 3.1, 6.1]]]])
        self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1.0e-6))

    def test_invalid_configuration_and_logit_contract_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ResidualCalibration(-1.0, 1.0, 0.0)
        base = torch.zeros(1, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "share shape"):
            apply_residual_calibration(
                base,
                torch.zeros(1, 1, 1, 1),
                CURRENT_ANCHOR,
            )
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            apply_residual_calibration(
                base,
                torch.full_like(base, float("nan")),
                CURRENT_ANCHOR,
            )


if __name__ == "__main__":
    unittest.main()
