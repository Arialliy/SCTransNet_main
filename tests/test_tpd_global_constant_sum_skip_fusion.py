from __future__ import annotations

import unittest

import torch

from model.tpd_global_constant_sum_skip_fusion import (
    FORMAL_GCSF_CHANNELS,
    GCSF_LOCAL_STATE_KEYS,
    GlobalConstantSumSkipFusion,
    PRODUCTION_GCSF_PARAMETERS,
    PRODUCTION_GCSF_STATE_KEY_COUNT,
    validate_formal_global_constant_sum_skip_fusion,
)


torch.set_num_threads(1)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


class GlobalConstantSumSkipFusionTests(unittest.TestCase):
    def test_formal_shape_state_manifest_and_hard_locks(self) -> None:
        module = GlobalConstantSumSkipFusion()
        manifest = validate_formal_global_constant_sum_skip_fusion(
            module,
            require_zero_initialization=True,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            PRODUCTION_GCSF_PARAMETERS,
        )
        self.assertEqual(len(module.state_dict()), PRODUCTION_GCSF_STATE_KEY_COUNT)
        self.assertEqual(tuple(module.state_dict()), GCSF_LOCAL_STATE_KEYS)
        self.assertEqual(tuple(module.named_buffers()), ())
        self.assertEqual(module.channels, FORMAL_GCSF_CHANNELS)
        self.assertEqual(module.gate_limit, 0.5)
        self.assertEqual(manifest["coefficient_sum"], 3.0)
        self.assertFalse(manifest["activation_norm_preserved"])

        invalid_channels = (
            (32.0, 64, 128, 256),
            (True, 64, 128, 256),
            (32, 64, 128),
            (32, 64, 0, 256),
        )
        for channels in invalid_channels:
            with self.subTest(channels=channels):
                with self.assertRaises((TypeError, ValueError)):
                    GlobalConstantSumSkipFusion(channels)
        for gate_limit in (0.25, 0.75, float("nan"), float("inf"), True):
            with self.subTest(gate_limit=gate_limit):
                with self.assertRaises((TypeError, ValueError)):
                    GlobalConstantSumSkipFusion(gate_limit=gate_limit)

    def test_zero_forward_and_shared_gradient_are_bitwise_anchored(self) -> None:
        module = GlobalConstantSumSkipFusion(dtype=torch.float64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260805)
        transformed = tuple(
            torch.randn(
                2,
                channels,
                3,
                5,
                generator=generator,
                dtype=torch.float64,
                requires_grad=True,
            )
            for channels in FORMAL_GCSF_CHANNELS
        )
        encoder = tuple(
            torch.randn(
                2,
                channels,
                3,
                5,
                generator=generator,
                dtype=torch.float64,
                requires_grad=True,
            )
            for channels in FORMAL_GCSF_CHANNELS
        )
        reference_t = tuple(value.detach().clone().requires_grad_() for value in transformed)
        reference_e = tuple(value.detach().clone().requires_grad_() for value in encoder)

        actual = module(transformed, encoder)
        expected = tuple(
            transformed_i.add(encoder_i).add(encoder_i)
            for transformed_i, encoder_i in zip(reference_t, reference_e)
        )
        for level, (actual_i, expected_i) in enumerate(zip(actual, expected)):
            with self.subTest(level=level, contract="forward"):
                self.assertTrue(torch.equal(_bits(actual_i), _bits(expected_i)))

        weights = tuple(
            torch.linspace(
                0.5,
                1.5,
                output.numel(),
                dtype=output.dtype,
            ).reshape_as(output)
            for output in actual
        )
        actual_loss = sum((output * weight).sum() for output, weight in zip(actual, weights))
        expected_loss = sum(
            (output * weight).sum() for output, weight in zip(expected, weights)
        )
        self.assertTrue(torch.equal(_bits(actual_loss), _bits(expected_loss)))
        actual_loss.backward()
        expected_loss.backward()
        for level in range(4):
            with self.subTest(level=level, contract="transformed_gradient"):
                self.assertTrue(
                    torch.equal(_bits(transformed[level].grad), _bits(reference_t[level].grad))
                )
            with self.subTest(level=level, contract="encoder_gradient"):
                self.assertTrue(
                    torch.equal(_bits(encoder[level].grad), _bits(reference_e[level].grad))
                )
            self.assertIsNotNone(module.reallocation_logits[level].grad)
            self.assertGreater(
                int(torch.count_nonzero(module.reallocation_logits[level].grad)),
                0,
            )

    def test_closed_bounds_finite_and_device_dtype_contract(self) -> None:
        module = GlobalConstantSumSkipFusion(dtype=torch.float32)
        with torch.no_grad():
            module.reallocation_logits[0].fill_(torch.finfo(torch.float32).max)
            module.reallocation_logits[1].fill_(-torch.finfo(torch.float32).max)
            module.reallocation_logits[2].fill_(1.25)
            module.reallocation_logits[3].fill_(-0.75)
        for level in range(4):
            transformed_coefficient, encoder_coefficient = module.coefficients(level)
            self.assertTrue(bool((transformed_coefficient >= 0.5).all()))
            self.assertTrue(bool((transformed_coefficient <= 1.5).all()))
            self.assertTrue(bool((encoder_coefficient >= 1.5).all()))
            self.assertTrue(bool((encoder_coefficient <= 2.5).all()))
            self.assertTrue(
                torch.equal(
                    transformed_coefficient + encoder_coefficient,
                    torch.full_like(transformed_coefficient, 3.0),
                )
            )

        transformed = torch.full((1, 32, 2, 2), 1.0e20)
        encoder = torch.full((1, 32, 2, 2), -2.0e19)
        output = module.forward_level(0, transformed, encoder)
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.sum().backward()
        self.assertTrue(
            bool(torch.isfinite(module.reallocation_logits[0].grad).all())
        )

        with self.assertRaisesRegex(TypeError, "share one dtype"):
            module.forward_level(0, transformed.double(), encoder)
        with self.assertRaisesRegex(ValueError, "equal shapes"):
            module.forward_level(0, transformed[:, :, :1], encoder)
        non_finite = transformed.clone()
        non_finite[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            module.forward_level(0, non_finite, encoder)
        with torch.no_grad():
            module.reallocation_logits[0][0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            module.gate(0)


if __name__ == "__main__":
    unittest.main()
