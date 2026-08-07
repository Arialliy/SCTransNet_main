from __future__ import annotations

import unittest

import torch

from experiments.pbdr_v4_component_loss import (
    _negative_component_loss,
    _positive_component_loss,
    compute_pbdr_v4_loss,
    role_loss_manifest,
)


def _ids() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rescue = torch.zeros(1, 1, 4, 4, dtype=torch.int64)
    suppress = torch.zeros_like(rescue)
    preserve = torch.zeros_like(rescue)
    rescue[0, 0, 0, 0:2] = 1
    rescue[0, 0, 1, 0] = 2
    suppress[0, 0, 3, 2:4] = 1
    preserve[0, 0, 2, 0:2] = 1
    return rescue, suppress, preserve


class PBDRV4ComponentLossTests(unittest.TestCase):
    def test_component_losses_are_equal_weight_per_component(self) -> None:
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        rescue, suppress, _ = _ids()
        positive = _positive_component_loss(logits, rescue)
        negative = _negative_component_loss(logits, suppress)
        self.assertAlmostEqual(
            float(positive.detach()),
            float(torch.log(torch.tensor(2.0))),
        )
        self.assertAlmostEqual(
            float(negative.detach()),
            float(torch.log(torch.tensor(2.0))),
        )
        (positive + negative).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertNotEqual(int(torch.count_nonzero(logits.grad)), 0)

    def test_roles_have_distinct_frozen_weights(self) -> None:
        miou = role_loss_manifest("best_miou")
        pd = role_loss_manifest("best_pd")
        self.assertNotEqual(miou["weights"], pd["weights"])
        self.assertGreater(pd["tversky_beta"], miou["tversky_beta"])
        self.assertIsNone(pd["performance_acceptance_margin"])

    def test_full_loss_is_finite_and_backpropagates(self) -> None:
        generator = torch.Generator().manual_seed(2026080701)
        base = torch.randn(1, 1, 4, 4, generator=generator)
        delta = torch.zeros_like(base, requires_grad=True)
        routed = base + delta
        target = torch.zeros_like(base)
        target[0, 0, 0, 0:2] = 1.0
        target[0, 0, 1, 0] = 1.0
        target[0, 0, 2, 0:2] = 1.0
        rescue, suppress, preserve = _ids()
        output = compute_pbdr_v4_loss(
            role="best_pd",
            routed_logits=routed,
            candidate_base_logits=base,
            reference_current_logits=base,
            delta_logits=delta,
            target=target,
            rescue_component_ids=rescue,
            suppress_component_ids=suppress,
            preserve_component_ids=preserve,
        )
        self.assertTrue(torch.isfinite(output.total))
        output.total.backward()
        self.assertIsNotNone(delta.grad)
        self.assertGreater(float(delta.grad.abs().sum()), 0.0)
        self.assertEqual(set(output.detached_scalars()), set(output.__dataclass_fields__))

    def test_empty_atlas_maps_are_supported(self) -> None:
        base = torch.zeros(1, 1, 2, 2)
        delta = torch.zeros_like(base, requires_grad=True)
        ids = torch.zeros(1, 1, 2, 2, dtype=torch.int32)
        output = compute_pbdr_v4_loss(
            role="best_miou",
            routed_logits=base + delta,
            candidate_base_logits=base,
            reference_current_logits=base,
            delta_logits=delta,
            target=torch.zeros_like(base),
            rescue_component_ids=ids,
            suppress_component_ids=ids,
            preserve_component_ids=ids,
        )
        self.assertEqual(float(output.rescue_components), 0.0)
        self.assertEqual(float(output.suppress_components), 0.0)
        self.assertEqual(float(output.preserve_components), 0.0)
        output.total.backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_contract_rejects_fractional_or_misaligned_ids(self) -> None:
        base = torch.zeros(1, 1, 2, 2)
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            compute_pbdr_v4_loss(
                role="best_miou",
                routed_logits=base,
                candidate_base_logits=base,
                reference_current_logits=base,
                delta_logits=base,
                target=base,
                rescue_component_ids=base,
                suppress_component_ids=base.to(torch.int64),
                preserve_component_ids=base.to(torch.int64),
            )

    def test_stage2_uses_independent_frozen_current_reference(self) -> None:
        ids = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
        target = torch.ones(1, 1, 2, 2)
        routed = torch.full_like(target, -2.0, requires_grad=True)
        current = torch.full_like(target, 2.0)
        moving_candidate_base = torch.full_like(target, -2.0)
        output = compute_pbdr_v4_loss(
            role="best_pd",
            routed_logits=routed,
            candidate_base_logits=moving_candidate_base,
            reference_current_logits=current,
            delta_logits=torch.zeros_like(target),
            target=target,
            rescue_component_ids=ids,
            suppress_component_ids=ids,
            preserve_component_ids=ids,
        )
        self.assertGreater(float(output.foreground_drop.detach()), 0.0)

        changed_candidate_base = torch.full_like(target, 20.0)
        changed = compute_pbdr_v4_loss(
            role="best_pd",
            routed_logits=routed,
            candidate_base_logits=changed_candidate_base,
            reference_current_logits=current,
            delta_logits=torch.zeros_like(target),
            target=target,
            rescue_component_ids=ids,
            suppress_component_ids=ids,
            preserve_component_ids=ids,
        )
        self.assertEqual(
            float(output.foreground_drop.detach()),
            float(changed.foreground_drop.detach()),
        )


if __name__ == "__main__":
    unittest.main()
