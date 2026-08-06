from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from model.tpd_persistent_evidence_residual_router_v2 import (
    FORMAL_CONFIDENCE_FLOOR,
    PBDR_V2_LOCAL_STATE_KEYS,
    PRODUCTION_PBDR_V2_PARAMETERS,
    PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
    PersistentEvidenceResidualRouterV2,
    validate_formal_pbdr_v2_router,
)


torch.set_num_threads(1)


class PersistentEvidenceResidualRouterV2Tests(unittest.TestCase):
    def test_counts_zero_state_rng_neutral_and_manifest(self) -> None:
        torch.manual_seed(2026080601)
        before = torch.get_rng_state().clone()
        router = PersistentEvidenceResidualRouterV2()
        after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(
            sum(parameter.numel() for parameter in router.parameters()),
            PRODUCTION_PBDR_V2_PARAMETERS,
        )
        self.assertEqual(
            len(router.state_dict()),
            PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
        )
        self.assertEqual(tuple(router.state_dict()), PBDR_V2_LOCAL_STATE_KEYS)
        self.assertEqual(len(tuple(router.buffers())), 0)
        for value in router.state_dict().values():
            self.assertEqual(int(torch.count_nonzero(value)), 0)
        manifest = validate_formal_pbdr_v2_router(
            router,
            require_zero_initialization=True,
        )
        self.assertEqual(manifest["parameters"], PRODUCTION_PBDR_V2_PARAMETERS)
        self.assertTrue(manifest["confidence_is_soft"])
        self.assertEqual(
            manifest["q4_gradient_boundary"],
            "stop_gradient_before_router",
        )

    def test_zero_anchor_soft_confidence_and_first_gradients(self) -> None:
        router = PersistentEvidenceResidualRouterV2()
        generator = torch.Generator().manual_seed(2026080602)
        q4 = torch.randn(2, 8, 4, 5, generator=generator, requires_grad=True)
        z_out = torch.randn(
            2, 1, 32, 40, generator=generator, requires_grad=True
        )
        z_d0 = torch.randn(
            2, 1, 32, 40, generator=generator, requires_grad=True
        )
        diagnostics = router.forward_with_diagnostics(z_out, z_d0, q4)
        self.assertTrue(torch.equal(diagnostics.routed_logits, z_out))
        self.assertTrue(
            torch.equal(
                diagnostics.confidence,
                torch.full_like(diagnostics.confidence, 0.5),
            )
        )
        self.assertGreater(
            float(diagnostics.confidence.detach().min()),
            FORMAL_CONFIDENCE_FLOOR - 1e-7,
        )
        self.assertLess(
            float(diagnostics.confidence.detach().max()),
            1.0 - FORMAL_CONFIDENCE_FLOOR + 1e-7,
        )
        diagnostics.routed_logits.square().mean().backward()
        self.assertIsNone(q4.grad)
        self.assertGreater(
            float(router.direct_residual_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            abs(float(router.rescue_strength_raw.grad)),
            0.0,
        )
        self.assertGreater(
            abs(float(router.suppression_strength_raw.grad)),
            0.0,
        )
        self.assertEqual(
            int(torch.count_nonzero(router.confidence_projection.weight.grad)),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(router.confidence_projection.bias.grad)),
            0,
        )

    def test_direct_rescue_and_suppression_paths_are_independent(self) -> None:
        q4 = torch.ones(1, 8, 2, 2)
        z_out = torch.zeros(1, 1, 8, 8)

        direct = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            direct.direct_residual_projection.weight[:, 0].fill_(0.5)
        direct_value = direct(z_out, z_out, q4)
        self.assertTrue(torch.all(direct_value > z_out))

        rescue = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            rescue.rescue_strength_raw.fill_(0.7)
        rescued = rescue(z_out, torch.ones_like(z_out), q4)
        self.assertTrue(torch.all(rescued > z_out))
        self.assertTrue(
            torch.equal(
                rescue(z_out, -torch.ones_like(z_out), q4),
                z_out,
            )
        )

        suppress = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            suppress.suppression_strength_raw.fill_(0.7)
        suppressed = suppress(z_out, -torch.ones_like(z_out), q4)
        self.assertTrue(torch.all(suppressed < z_out))
        self.assertTrue(
            torch.equal(
                suppress(z_out, torch.ones_like(z_out), q4),
                z_out,
            )
        )

    def test_confidence_learns_after_route_leaves_zero_anchor(self) -> None:
        router = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            router.direct_residual_projection.weight[:, 0].fill_(0.25)
        q4 = torch.zeros(1, 8, 3, 3)
        q4[:, 0, 1, 1] = 2.0
        z_out = torch.zeros(1, 1, 12, 12)
        value = router(z_out, z_out, q4)
        value.square().mean().backward()
        confidence_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in router.confidence_projection.parameters()
        )
        self.assertGreater(confidence_gradient, 0.0)

    def test_nonzero_diagnostics_match_reference_formula(self) -> None:
        router = PersistentEvidenceResidualRouterV2()
        with torch.no_grad():
            router.confidence_projection.weight.copy_(
                torch.linspace(-0.3, 0.4, 8).reshape(1, 8, 1, 1)
            )
            router.confidence_projection.bias.fill_(0.37)
            router.direct_residual_projection.weight.copy_(
                torch.linspace(0.25, -0.15, 8).reshape(1, 8, 1, 1)
            )
            router.rescue_strength_raw.fill_(0.6)
            router.suppression_strength_raw.fill_(0.9)
        generator = torch.Generator().manual_seed(2026080606)
        q4 = torch.randn(2, 8, 3, 4, generator=generator)
        z_out = torch.randn(2, 1, 9, 10, generator=generator)
        z_d0 = torch.randn(2, 1, 9, 10, generator=generator)
        actual = router.forward_with_diagnostics(z_out, z_d0, q4)

        rms = q4.detach().square().mean((1, 2, 3), keepdim=True).sqrt()
        normalized = q4.detach() / rms.clamp_min(1.0e-6)
        confidence_logits = F.conv2d(
            normalized,
            router.confidence_projection.weight,
            router.confidence_projection.bias,
        )
        direct_logits = F.conv2d(
            normalized,
            router.direct_residual_projection.weight,
        )
        confidence_logits = F.interpolate(
            confidence_logits,
            size=z_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        direct_logits = F.interpolate(
            direct_logits,
            size=z_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence = 0.05 + 0.90 * torch.sigmoid(confidence_logits)
        direct_residual = confidence * torch.tanh(direct_logits)
        disagreement = z_d0 - z_out
        target_rescue = confidence * F.relu(disagreement)
        background_suppression = (1.0 - confidence) * F.relu(-disagreement)
        rescue_strength = 0.5 * torch.tanh(router.rescue_strength_raw)
        suppression_strength = 0.5 * torch.tanh(
            router.suppression_strength_raw
        )
        routed = (
            z_out
            + direct_residual
            + rescue_strength.view(1, 1, 1, 1) * target_rescue
            - suppression_strength.view(1, 1, 1, 1)
            * background_suppression
        )
        for observed, expected in (
            (actual.confidence, confidence),
            (actual.direct_residual, direct_residual),
            (actual.target_rescue, target_rescue),
            (actual.background_suppression, background_suppression),
            (actual.routed_logits, routed),
        ):
            torch.testing.assert_close(
                observed,
                expected,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        self.assertFalse(
            torch.equal(actual.confidence, torch.full_like(confidence, 0.5))
        )

    def test_invalid_contracts_and_nonzero_validator(self) -> None:
        router = PersistentEvidenceResidualRouterV2()
        q4 = torch.zeros(1, 8, 2, 2)
        out = torch.zeros(1, 1, 8, 8)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            router(out, torch.zeros(1, 1, 7, 8), q4)
        with self.assertRaisesRegex(ValueError, "C=8"):
            router(out, out, torch.zeros(1, 7, 2, 2))
        with self.assertRaisesRegex(FloatingPointError, "q4"):
            invalid = q4.clone()
            invalid[0, 0, 0, 0] = float("nan")
            router(out, out, invalid)

        trained = copy.deepcopy(router)
        with torch.no_grad():
            trained.rescue_strength_raw.fill_(0.1)
        validate_formal_pbdr_v2_router(
            trained,
            require_zero_initialization=False,
        )
        with self.assertRaisesRegex(RuntimeError, "not zero"):
            validate_formal_pbdr_v2_router(
                trained,
                require_zero_initialization=True,
            )


if __name__ == "__main__":
    unittest.main()
