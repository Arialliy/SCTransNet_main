from __future__ import annotations

import inspect
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_role_aligned_residual_calibrator_v4 import (
    PBDR_V4_LOCAL_STATE_KEYS,
    PRODUCTION_PBDR_V4_BUFFER_COUNT,
    PRODUCTION_PBDR_V4_PARAMETERS,
    PRODUCTION_PBDR_V4_STATE_KEY_COUNT,
    RoleAlignedResidualCalibratorV4,
    validate_formal_pbdr_v4_calibrator,
)


torch.set_num_threads(1)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _inputs(
    *,
    seed: int = 2026080701,
    batch: int = 2,
    height: int = 8,
    width: int = 8,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "z_out": torch.randn(batch, 1, height, width, generator=generator),
        "z_d0": torch.randn(batch, 1, height, width, generator=generator),
        "z_gt2": torch.randn(batch, 1, height, width, generator=generator),
        "z_gt3": torch.randn(batch, 1, height, width, generator=generator),
        "z_gt4": torch.randn(batch, 1, height, width, generator=generator),
        "z_gt5": torch.randn(batch, 1, height, width, generator=generator),
        "q4": torch.randn(batch, 8, 3, 3, generator=generator),
        "local_feature": torch.randn(
            batch,
            32,
            height,
            width,
            generator=generator,
        ),
    }


class RoleAlignedResidualCalibratorV4Tests(unittest.TestCase):
    def test_formal_counts_named_auxiliary_contract_and_manifest(self) -> None:
        signature = inspect.signature(
            RoleAlignedResidualCalibratorV4.forward_with_diagnostics
        )
        self.assertNotIn("auxiliary_logits", signature.parameters)
        for name in ("z_gt2", "z_gt3", "z_gt4", "z_gt5"):
            self.assertIn(name, signature.parameters)

        for role in ("best_miou", "best_pd"):
            calibrator = RoleAlignedResidualCalibratorV4(role=role)
            manifest = validate_formal_pbdr_v4_calibrator(
                calibrator,
                expected_role=role,
                require_identity_initialization=True,
            )
            self.assertEqual(
                _parameter_count(calibrator),
                PRODUCTION_PBDR_V4_PARAMETERS,
            )
            self.assertEqual(
                len(calibrator.state_dict()),
                PRODUCTION_PBDR_V4_STATE_KEY_COUNT,
            )
            self.assertEqual(
                tuple(calibrator.state_dict()),
                PBDR_V4_LOCAL_STATE_KEYS,
            )
            self.assertEqual(
                len(tuple(calibrator.buffers())),
                PRODUCTION_PBDR_V4_BUFFER_COUNT,
            )
            self.assertEqual(calibrator.context_stem[0].in_channels, 38)
            self.assertEqual(manifest["scalar_context_channels"], 14)
            self.assertEqual(
                manifest["auxiliary_logit_order"],
                ("gt2", "gt3", "gt4", "gt5"),
            )
            self.assertEqual(
                manifest["persistent_buffer_names"],
                ("role_code", "positive_limit", "negative_limit"),
            )

    def test_both_roles_are_exact_identity_for_finite_inputs(self) -> None:
        inputs = _inputs()
        for role in ("best_miou", "best_pd"):
            calibrator = RoleAlignedResidualCalibratorV4(role=role)
            routing = calibrator.forward_with_diagnostics(**inputs)
            self.assertTrue(torch.equal(routing.routed_logits, inputs["z_out"]))
            self.assertEqual(int(torch.count_nonzero(routing.delta_logits)), 0)
            self.assertEqual(int(torch.count_nonzero(routing.signed_score)), 0)
            self.assertTrue(bool((routing.rescue_budget >= 0.0).all()))
            self.assertTrue(bool((routing.rescue_budget <= 1.0).all()))
            self.assertTrue(bool((routing.suppression_budget >= 0.0).all()))
            self.assertTrue(bool((routing.suppression_budget <= 1.0).all()))

    def test_cross_role_and_different_limit_loads_fail_even_non_strict(self) -> None:
        target = RoleAlignedResidualCalibratorV4(role="best_miou")
        source_role = RoleAlignedResidualCalibratorV4(role="best_pd")
        with self.assertRaisesRegex(RuntimeError, "role_code"):
            target.load_state_dict(source_role.state_dict(), strict=False)

        target = RoleAlignedResidualCalibratorV4(role="best_miou")
        source_positive = RoleAlignedResidualCalibratorV4(
            role="best_miou",
            positive_limit=0.61,
        )
        with self.assertRaisesRegex(RuntimeError, "positive_limit"):
            target.load_state_dict(source_positive.state_dict(), strict=False)

        target = RoleAlignedResidualCalibratorV4(role="best_miou")
        source_negative = RoleAlignedResidualCalibratorV4(
            role="best_miou",
            negative_limit=0.49,
        )
        with self.assertRaisesRegex(RuntimeError, "negative_limit"):
            target.load_state_dict(source_negative.state_dict(), strict=False)

        missing_semantics = dict(target.state_dict())
        del missing_semantics["role_code"]
        with self.assertRaisesRegex(RuntimeError, "role_code"):
            target.load_state_dict(missing_semantics, strict=False)

        same_semantics = RoleAlignedResidualCalibratorV4(role="best_miou")
        result = target.load_state_dict(same_semantics.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])

    def test_first_step_head_then_second_step_upstream_gradients(self) -> None:
        torch.manual_seed(2026080702)
        calibrator = RoleAlignedResidualCalibratorV4(role="best_pd")
        inputs = _inputs(seed=2026080703)
        inputs["z_out"] = torch.zeros_like(inputs["z_out"])
        target = torch.zeros_like(inputs["z_out"])
        optimizer = torch.optim.SGD(calibrator.parameters(), lr=0.05)

        first = calibrator.forward_with_diagnostics(**inputs)
        first_loss = F.binary_cross_entropy_with_logits(
            first.routed_logits,
            target,
        )
        first_loss.backward()
        head_weight_gradient = calibrator.residual_head.weight.grad
        head_bias_gradient = calibrator.residual_head.bias.grad
        self.assertIsNotNone(head_weight_gradient)
        self.assertIsNotNone(head_bias_gradient)
        assert head_weight_gradient is not None
        assert head_bias_gradient is not None
        self.assertTrue(bool(torch.isfinite(head_weight_gradient).all()))
        self.assertTrue(bool(torch.isfinite(head_bias_gradient).all()))
        self.assertEqual(
            int(torch.count_nonzero(head_weight_gradient[0, :, 0, 0])),
            24,
        )
        self.assertGreater(float(head_bias_gradient.abs().sum()), 0.0)

        upstream_first = [
            parameter.grad
            for name, parameter in calibrator.named_parameters()
            if not name.startswith("residual_head.")
        ]
        self.assertTrue(all(gradient is not None for gradient in upstream_first))
        self.assertEqual(
            sum(
                int(torch.count_nonzero(gradient))
                for gradient in upstream_first
                if gradient is not None
            ),
            0,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        second = calibrator.forward_with_diagnostics(**inputs)
        second_loss = F.binary_cross_entropy_with_logits(
            second.routed_logits,
            target,
        )
        second_loss.backward()
        upstream_second = [
            parameter.grad
            for name, parameter in calibrator.named_parameters()
            if not name.startswith("residual_head.")
        ]
        self.assertTrue(
            all(
                gradient is None or bool(torch.isfinite(gradient).all())
                for gradient in upstream_second
            )
        )
        self.assertGreater(
            sum(
                int(torch.count_nonzero(gradient))
                for gradient in upstream_second
                if gradient is not None
            ),
            0,
        )

    def test_saturated_false_positive_uses_suppression_subgradient(self) -> None:
        torch.manual_seed(2026080704)
        calibrator = RoleAlignedResidualCalibratorV4(role="best_miou")
        inputs = _inputs(seed=2026080705, batch=1)
        inputs["z_out"] = torch.full_like(inputs["z_out"], 100.0)
        inputs["z_d0"] = torch.full_like(inputs["z_d0"], -100.0)
        for name in ("z_gt2", "z_gt3", "z_gt4", "z_gt5"):
            inputs[name] = torch.full_like(inputs[name], -100.0)

        initial = calibrator.forward_with_diagnostics(**inputs)
        self.assertTrue(torch.equal(initial.routed_logits, inputs["z_out"]))
        self.assertEqual(int(torch.count_nonzero(initial.rescue_budget)), 0)
        self.assertTrue(torch.equal(
            initial.suppression_budget,
            torch.ones_like(initial.suppression_budget),
        ))
        loss = F.binary_cross_entropy_with_logits(
            initial.routed_logits,
            torch.zeros_like(initial.routed_logits),
        )
        loss.backward()
        bias_gradient = calibrator.residual_head.bias.grad
        self.assertIsNotNone(bias_gradient)
        assert bias_gradient is not None
        self.assertGreater(float(bias_gradient), 0.0)

        with torch.no_grad():
            calibrator.residual_head.weight.zero_()
            calibrator.residual_head.bias.sub_(0.1 * bias_gradient)
        suppressed = calibrator.forward_with_diagnostics(**inputs)
        self.assertTrue(bool((suppressed.signed_score < 0.0).all()))
        self.assertTrue(bool((suppressed.delta_logits < 0.0).all()))
        self.assertTrue(bool((suppressed.routed_logits < inputs["z_out"]).all()))
        self.assertLessEqual(
            float(suppressed.delta_logits.detach().abs().max()),
            float(calibrator.negative_limit.detach()),
        )

    def test_shape_dtype_and_optional_finite_validation(self) -> None:
        calibrator = RoleAlignedResidualCalibratorV4(role="best_pd")
        inputs = _inputs()

        bad_channels = dict(inputs)
        bad_channels["q4"] = inputs["q4"][:, :7]
        with self.assertRaisesRegex(ValueError, "q4 requires C=8"):
            calibrator.forward_with_diagnostics(**bad_channels)

        bad_shape = dict(inputs)
        bad_shape["z_gt3"] = inputs["z_gt3"][:, :, :-1]
        with self.assertRaisesRegex(ValueError, "must match z_out"):
            calibrator.forward_with_diagnostics(**bad_shape)

        bad_dtype = dict(inputs)
        bad_dtype["z_gt4"] = inputs["z_gt4"].double()
        with self.assertRaisesRegex(ValueError, "z_gt4 dtype must match z_out"):
            calibrator.forward_with_diagnostics(**bad_dtype)

        debug = RoleAlignedResidualCalibratorV4(
            role="best_pd",
            debug_validate_finite=True,
        )
        non_finite = _inputs()
        non_finite["z_out"] = non_finite["z_out"].clone()
        non_finite["z_out"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "z_out"):
            debug.forward_with_diagnostics(**non_finite)


if __name__ == "__main__":
    unittest.main()
