from __future__ import annotations

import unittest

import torch

from experiments.pbdr_v4_component_loss import (
    ROLE_WEIGHTS,
    compute_pbdr_v4_loss,
)
from experiments.pbdr_v5_target_preservation_loss import (
    _active_background_probability_increase,
    _component_peak_no_drop,
    _component_positive_support_logit_no_drop,
    compute_pbdr_v5_target_preservation_loss,
    target_preservation_loss_manifest,
)


def _empty_ids(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int64)


def _compute(
    *,
    routed: torch.Tensor,
    current: torch.Tensor,
    target: torch.Tensor,
    preserve: torch.Tensor,
    rescue: torch.Tensor | None = None,
    suppress: torch.Tensor | None = None,
    role: str = "best_miou",
):
    zero_ids = _empty_ids(tuple(routed.shape))
    return compute_pbdr_v5_target_preservation_loss(
        role=role,  # type: ignore[arg-type]
        routed_logits=routed,
        candidate_base_logits=current.detach().clone(),
        reference_current_logits=current,
        delta_logits=routed - current.detach(),
        target=target,
        rescue_component_ids=zero_ids if rescue is None else rescue,
        suppress_component_ids=zero_ids if suppress is None else suppress,
        preserve_component_ids=preserve,
    )


class PBDRV5TargetPreservationLossTests(unittest.TestCase):
    def test_manifest_freezes_one_to_one_replacements_and_no_margin(self) -> None:
        manifest = target_preservation_loss_manifest("best_pd")
        self.assertEqual(manifest["weights"], ROLE_WEIGHTS["best_pd"])
        self.assertEqual(
            set(manifest["replacements"]),
            {"preserve", "foreground_drop", "background_increase"},
        )
        self.assertEqual(manifest["fixed_probability_comparison"], ">")
        self.assertEqual(manifest["fixed_probability_threshold"], 0.5)
        self.assertIsNone(manifest["performance_acceptance_margin"])
        self.assertEqual(manifest["current_reference_gradient"], "detached")
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            target_preservation_loss_manifest("unknown")  # type: ignore[arg-type]

    def test_empty_component_maps_have_exact_zero_replacements(self) -> None:
        current = torch.zeros(2, 1, 3, 4, requires_grad=True)
        routed = torch.zeros_like(current, requires_grad=True)
        target = torch.zeros_like(current)
        preserve = _empty_ids(tuple(current.shape))
        output = _compute(
            routed=routed,
            current=current,
            target=target,
            preserve=preserve,
        )
        self.assertEqual(float(output.preserve_peak_no_drop.detach()), 0.0)
        self.assertEqual(
            float(output.preserve_positive_support_logit_no_drop.detach()),
            0.0,
        )
        self.assertEqual(float(output.active_background_increase.detach()), 0.0)
        self.assertTrue(torch.isfinite(output.total))
        self.assertEqual(
            set(output.detached_scalars()),
            set(output.__dataclass_fields__),
        )
        output.total.backward()
        self.assertIsNotNone(routed.grad)
        self.assertIsNone(current.grad)

    def test_equal_current_and_candidate_is_zero_margin_no_drop(self) -> None:
        current = torch.tensor(
            [[[[2.0, 1.0, -1.0], [0.5, -2.0, -3.0]]]],
            requires_grad=True,
        )
        routed = current.detach().clone().requires_grad_(True)
        preserve = _empty_ids(tuple(current.shape))
        preserve[0, 0, 0, 0:2] = 1
        target = torch.zeros_like(current)
        target[preserve > 0] = 1.0
        output = _compute(
            routed=routed,
            current=current,
            target=target,
            preserve=preserve,
        )
        self.assertEqual(float(output.preserve_peak_no_drop.detach()), 0.0)
        self.assertEqual(
            float(output.preserve_positive_support_logit_no_drop.detach()),
            0.0,
        )
        self.assertEqual(float(output.active_background_increase.detach()), 0.0)

    def test_peak_and_positive_support_drops_are_penalized_only_downward(self) -> None:
        current = torch.full((1, 1, 2, 3), 2.0)
        preserve = _empty_ids(tuple(current.shape))
        preserve[0, 0, 0, 0:2] = 1
        dropped = current.clone()
        dropped[preserve > 0] = 1.0
        improved = current.clone()
        improved[preserve > 0] = 3.0

        peak_drop = _component_peak_no_drop(dropped, current, preserve)
        support_drop = _component_positive_support_logit_no_drop(
            dropped,
            current,
            preserve,
        )
        self.assertAlmostEqual(float(peak_drop), 1.0, places=6)
        self.assertAlmostEqual(float(support_drop), 1.0, places=6)
        self.assertEqual(
            float(_component_peak_no_drop(improved, current, preserve)),
            0.0,
        )
        self.assertEqual(
            float(
                _component_positive_support_logit_no_drop(
                    improved,
                    current,
                    preserve,
                )
            ),
            0.0,
        )

    def test_reference_is_frozen_and_gradients_reach_only_candidate(self) -> None:
        current = torch.tensor(
            [[[[2.0, 2.0], [0.0, 0.0]]]],
            requires_grad=True,
        )
        routed = torch.tensor(
            [[[[1.0, 1.0], [1.0, -1.0]]]],
            requires_grad=True,
        )
        preserve = _empty_ids(tuple(current.shape))
        preserve[0, 0, 0, 0:2] = 1
        target = torch.zeros_like(current)
        target[preserve > 0] = 1.0
        output = _compute(
            routed=routed,
            current=current,
            target=target,
            preserve=preserve,
            role="best_pd",
        )
        output.total.backward()
        self.assertIsNotNone(routed.grad)
        self.assertGreater(float(routed.grad.abs().sum()), 0.0)
        self.assertIsNone(current.grad)

    def test_preserve_reductions_are_equal_per_component_not_per_pixel(self) -> None:
        current = torch.full((1, 1, 1, 5), 2.0)
        candidate = current.clone()
        preserve = _empty_ids(tuple(current.shape))
        preserve[0, 0, 0, 0] = 1
        preserve[0, 0, 0, 1:5] = 2
        candidate[0, 0, 0, 0] = 0.0

        # Component 1 contributes 2^2=4 and component 2 contributes zero,
        # hence equal-component reduction is 2 rather than pixel mean 0.8.
        self.assertAlmostEqual(
            float(_component_peak_no_drop(candidate, current, preserve)),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(
                _component_positive_support_logit_no_drop(
                    candidate,
                    current,
                    preserve,
                )
            ),
            2.0,
            places=6,
        )

    def test_active_background_is_not_diluted_by_unchanged_pixels(self) -> None:
        current_small = torch.zeros(1, 1, 1, 1)
        candidate_small = torch.ones_like(current_small)
        target_small = torch.zeros_like(current_small)
        small = _active_background_probability_increase(
            candidate_small,
            current_small,
            target_small,
        )

        current_large = torch.zeros(1, 1, 1, 100)
        candidate_large = torch.zeros_like(current_large)
        candidate_large[0, 0, 0, 0] = 1.0
        candidate_large[0, 0, 0, 1] = -1.0
        target_large = torch.zeros_like(current_large)
        large = _active_background_probability_increase(
            candidate_large,
            current_large,
            target_large,
        )
        expected = (torch.sigmoid(torch.tensor(1.0)) - 0.5).square()
        self.assertAlmostEqual(float(small), float(expected), places=7)
        self.assertAlmostEqual(float(large), float(expected), places=7)

        # Reduction is per sample: a second sample with no active increase
        # contributes zero instead of changing the active-pixel denominator.
        current_batch = torch.zeros(2, 1, 1, 100)
        candidate_batch = torch.zeros_like(current_batch)
        candidate_batch[0, 0, 0, 0] = 1.0
        target_batch = torch.zeros_like(current_batch)
        batched = _active_background_probability_increase(
            candidate_batch,
            current_batch,
            target_batch,
        )
        self.assertAlmostEqual(float(batched), float(expected / 2.0), places=7)

    def test_total_is_exact_v4_one_to_one_replacement(self) -> None:
        current = torch.tensor(
            [[[[2.0, 2.0], [0.0, 0.0]]]],
        )
        routed = torch.tensor(
            [[[[1.0, 1.0], [1.0, 0.0]]]],
            requires_grad=True,
        )
        target = torch.tensor(
            [[[[1.0, 1.0], [0.0, 0.0]]]],
        )
        preserve = _empty_ids(tuple(current.shape))
        preserve[0, 0, 0, 0:2] = 1
        zero_ids = _empty_ids(tuple(current.shape))
        delta = routed - current
        v4 = compute_pbdr_v4_loss(
            role="best_miou",
            routed_logits=routed,
            candidate_base_logits=current,
            reference_current_logits=current,
            delta_logits=delta,
            target=target,
            rescue_component_ids=zero_ids,
            suppress_component_ids=zero_ids,
            preserve_component_ids=preserve,
        )
        v5 = compute_pbdr_v5_target_preservation_loss(
            role="best_miou",
            routed_logits=routed,
            candidate_base_logits=current,
            reference_current_logits=current,
            delta_logits=delta,
            target=target,
            rescue_component_ids=zero_ids,
            suppress_component_ids=zero_ids,
            preserve_component_ids=preserve,
        )
        weights = ROLE_WEIGHTS["best_miou"]
        expected = (
            v4.total
            - weights["preserve"] * v4.preserve_components
            - weights["foreground_drop"] * v4.foreground_drop
            - weights["background_increase"] * v4.background_increase
            + weights["preserve"] * v5.preserve_peak_no_drop
            + weights["foreground_drop"]
            * v5.preserve_positive_support_logit_no_drop
            + weights["background_increase"] * v5.active_background_increase
        )
        torch.testing.assert_close(v5.total, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(v5.bce, v4.bce, rtol=0.0, atol=0.0)
        torch.testing.assert_close(v5.tversky, v4.tversky, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            v5.rescue_components,
            v4.rescue_components,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            v5.suppress_components,
            v4.suppress_components,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            v5.neutral_delta,
            v4.neutral_delta,
            rtol=0.0,
            atol=0.0,
        )

    def test_compute_reuses_v4_input_validation(self) -> None:
        base = torch.zeros(1, 1, 2, 2)
        ids = _empty_ids(tuple(base.shape))
        common = {
            "role": "best_miou",
            "routed_logits": base,
            "candidate_base_logits": base,
            "reference_current_logits": base,
            "delta_logits": base,
            "target": base,
            "rescue_component_ids": ids,
            "suppress_component_ids": ids,
            "preserve_component_ids": ids,
        }
        fractional = dict(common)
        fractional["preserve_component_ids"] = base
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            compute_pbdr_v5_target_preservation_loss(**fractional)  # type: ignore[arg-type]

        wrong_shape = dict(common)
        wrong_shape["reference_current_logits"] = torch.zeros(1, 1, 1, 2)
        with self.assertRaisesRegex(ValueError, "share the routed-logit shape"):
            compute_pbdr_v5_target_preservation_loss(**wrong_shape)  # type: ignore[arg-type]

        invalid_target = dict(common)
        invalid_target["target"] = torch.full_like(base, 1.5)
        with self.assertRaisesRegex(ValueError, r"target must lie in \[0, 1\]"):
            compute_pbdr_v5_target_preservation_loss(**invalid_target)  # type: ignore[arg-type]

        non_finite = dict(common)
        non_finite["routed_logits"] = torch.full_like(base, float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            compute_pbdr_v5_target_preservation_loss(**non_finite)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
