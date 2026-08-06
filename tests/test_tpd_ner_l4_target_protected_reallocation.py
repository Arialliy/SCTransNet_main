from __future__ import annotations

import copy
import unittest

import torch

from model.tpd_ner_l4_target_protected_reallocation import (
    NERL4TargetProtectedReallocation,
    validate_formal_ner_l4_target_protected_reallocation,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
    build_formal_v4_qfg_v2_croa_l4_tpr_survival_model,
    load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _branches(
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026080503)
    transformed = torch.randn(
        2,
        256,
        5,
        7,
        generator=generator,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    encoder = torch.randn(
        2,
        256,
        5,
        7,
        generator=generator,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    return transformed, encoder


def _background_q4(
    *,
    dtype: torch.dtype = torch.float64,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.zeros(
        2,
        8,
        5,
        7,
        dtype=dtype,
        requires_grad=requires_grad,
    )


class NERL4TargetProtectedReallocationCoreTests(unittest.TestCase):
    def test_formal_state_and_frozen_constants(self) -> None:
        module = NERL4TargetProtectedReallocation()
        manifest = validate_formal_ner_l4_target_protected_reallocation(
            module,
            require_zero_initialization=True,
        )
        self.assertEqual(tuple(module.state_dict()), ("reallocation_logits",))
        self.assertEqual(tuple(module.named_buffers()), ())
        self.assertEqual(tuple(module.reallocation_logits.shape), (1, 256, 1, 1))
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            256,
        )
        self.assertEqual(int(torch.count_nonzero(module.reallocation_logits)), 0)
        self.assertEqual(module.gate_limit, 0.25)
        self.assertEqual(module.tail_z_threshold, 1.5)
        self.assertEqual(module.dilation_kernel, 3)
        self.assertEqual(manifest["protected_region_fusion"], "(T+E)+E")
        self.assertEqual(manifest["coefficient_sum"], 3.0)
        self.assertTrue(manifest["coefficient_sum_is_constant"])
        self.assertTrue(manifest["protection_detached"])
        gate = module.gate()
        self.assertEqual(tuple(gate.shape), (1, 256, 1, 1))
        self.assertEqual(int(torch.count_nonzero(gate)), 0)

    def test_zero_gate_is_bitwise_current_l4_fusion_and_shared_gradient(self) -> None:
        module = NERL4TargetProtectedReallocation().to(dtype=torch.float64)
        transformed, encoder = _branches(requires_grad=True)
        q4 = _background_q4(requires_grad=True)
        reference_t = transformed.detach().clone().requires_grad_()
        reference_e = encoder.detach().clone().requires_grad_()

        actual = module(transformed, encoder, q4)
        expected = reference_t.add(reference_e).add(reference_e)
        self.assertTrue(torch.equal(_bits(actual), _bits(expected)))

        weights = torch.linspace(
            0.25,
            1.75,
            actual.numel(),
            dtype=actual.dtype,
        ).reshape_as(actual)
        actual_loss = actual.mul(weights).sum()
        expected_loss = expected.mul(weights).sum()
        self.assertTrue(torch.equal(_bits(actual_loss), _bits(expected_loss)))
        actual_loss.backward()
        expected_loss.backward()

        self.assertTrue(torch.equal(_bits(transformed.grad), _bits(reference_t.grad)))
        self.assertTrue(torch.equal(_bits(encoder.grad), _bits(reference_e.grad)))
        self.assertIsNone(q4.grad)
        self.assertIsNotNone(module.reallocation_logits.grad)
        self.assertTrue(bool(torch.isfinite(module.reallocation_logits.grad).all()))
        self.assertGreater(
            int(torch.count_nonzero(module.reallocation_logits.grad)),
            0,
        )

    def test_target_zone_is_strict_original_and_background_uses_frozen_formula(self) -> None:
        module = NERL4TargetProtectedReallocation().to(dtype=torch.float64)
        with torch.no_grad():
            module.reallocation_logits.fill_(0.8)
        transformed, encoder = _branches()
        q4 = _background_q4()
        q4[:, :, 2, 3] = 100.0

        protection = module.build_protection(q4)
        self.assertEqual(tuple(protection.shape), (2, 1, 5, 7))
        self.assertEqual(protection.dtype, q4.dtype)
        self.assertEqual(protection.device, q4.device)
        self.assertFalse(protection.requires_grad)
        self.assertEqual(
            set(float(value) for value in torch.unique(protection)),
            {0.0, 1.0},
        )
        self.assertEqual(
            int(torch.count_nonzero(protection[0, 0, 1:4, 2:5])),
            9,
        )

        actual = module(transformed, encoder, q4)
        baseline = transformed.add(encoder).add(encoder)
        gate = module.gate()
        background = protection.new_tensor(1.0).sub(protection)
        effective_gate = background.mul(gate)
        expected = baseline.add(
            effective_gate.mul(transformed).sub(
                effective_gate.mul(encoder)
            )
        )
        self.assertTrue(torch.equal(_bits(actual), _bits(expected)))

        protected = protection.bool().expand_as(actual)
        self.assertTrue(
            torch.equal(_bits(actual[protected]), _bits(baseline[protected]))
        )
        unprotected = torch.logical_not(protected)
        self.assertGreater(int(torch.count_nonzero(unprotected)), 0)
        self.assertTrue(
            torch.equal(_bits(actual[unprotected]), _bits(expected[unprotected]))
        )

        transformed_coefficient = effective_gate.add(1.0)
        encoder_coefficient = effective_gate.neg().add(2.0)
        self.assertTrue(
            torch.equal(
                transformed_coefficient + encoder_coefficient,
                torch.full_like(transformed_coefficient, 3.0),
            )
        )
        self.assertTrue(
            torch.equal(
                transformed_coefficient[protected],
                torch.ones_like(transformed_coefficient[protected]),
            )
        )
        self.assertTrue(
            torch.equal(
                encoder_coefficient[protected],
                torch.full_like(encoder_coefficient[protected], 2.0),
            )
        )

    def test_protection_route_is_detached_but_gate_remains_trainable(self) -> None:
        module = NERL4TargetProtectedReallocation().to(dtype=torch.float64)
        transformed, encoder = _branches(requires_grad=True)
        q4 = _background_q4(requires_grad=True)
        q4_data = q4.detach()
        q4_data[:, :, 2, 3] = 100.0
        output = module(transformed, encoder, q4)
        output.square().mean().backward()

        self.assertIsNone(q4.grad)
        self.assertIsNotNone(transformed.grad)
        self.assertIsNotNone(encoder.grad)
        self.assertIsNotNone(module.reallocation_logits.grad)
        self.assertGreater(
            int(torch.count_nonzero(module.reallocation_logits.grad)),
            0,
        )

    def test_shape_dtype_device_and_finite_contract(self) -> None:
        module = NERL4TargetProtectedReallocation()
        transformed, encoder = _branches(dtype=torch.float32)
        q4 = _background_q4(dtype=torch.float32)
        output = module(transformed, encoder, q4)
        self.assertEqual(output.shape, transformed.shape)
        self.assertEqual(output.dtype, transformed.dtype)
        self.assertEqual(output.device, transformed.device)
        self.assertTrue(bool(torch.isfinite(output).all()))

        with self.assertRaises((TypeError, ValueError)):
            module(transformed[:, :255], encoder[:, :255], q4)
        with self.assertRaises((TypeError, ValueError)):
            module(transformed[:, :, :4], encoder, q4)
        with self.assertRaises((TypeError, ValueError)):
            module(transformed, encoder.double(), q4)
        with self.assertRaises((TypeError, ValueError)):
            module(transformed, encoder, q4[:, :, :4])
        with self.assertRaises((TypeError, ValueError)):
            module(transformed, encoder, q4.double())

        nonfinite_branch = transformed.clone()
        nonfinite_branch[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            module(nonfinite_branch, encoder, q4)
        nonfinite_q4 = q4.clone()
        nonfinite_q4[0, 0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            module(transformed, encoder, nonfinite_q4)
        with torch.no_grad():
            module.reallocation_logits[0, 0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            module.gate()


class NERL4TargetProtectedReallocationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.candidate, _ = build_formal_v4_qfg_v2_croa_l4_tpr_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.candidate

    def test_strict_current_state_migration_adds_only_the_zero_gate(self) -> None:
        source = {
            name: value.detach().clone()
            for name, value in self.current.state_dict().items()
        }
        source_keys = set(source)
        target_keys = set(self.candidate.state_dict())
        self.assertEqual(
            target_keys - source_keys,
            {"ner_l4_tpr.reallocation_logits"},
        )
        self.assertEqual(source_keys - target_keys, set())

        report = load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension(
            self.candidate,
            source,
        )
        self.assertEqual(report["new_state_key_count"], 1)
        self.assertEqual(
            tuple(report["new_state_keys"]),
            ("ner_l4_tpr.reallocation_logits",),
        )
        target = self.candidate.state_dict()
        for name, expected in source.items():
            with self.subTest(shared_state=name):
                self.assertTrue(torch.equal(target[name], expected))
        self.assertEqual(
            int(torch.count_nonzero(target["ner_l4_tpr.reallocation_logits"])),
            0,
        )

        missing = dict(source)
        missing.pop(next(iter(missing)))
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension(
                copy.deepcopy(self.candidate),
                missing,
            )
        foreign = dict(source)
        foreign["fabricated"] = torch.zeros(1)
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension(
                copy.deepcopy(self.candidate),
                foreign,
            )

    def test_zero_gate_integrated_model_is_bitwise_current_final(self) -> None:
        self.assertIs(
            type(self.candidate),
            TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
        )
        self.current.eval()
        self.candidate.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080504)
        images = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            expected = self.current(images)
            actual = self.candidate(images)
        self.assertEqual(len(actual), len(expected))
        for index, (actual_i, expected_i) in enumerate(zip(actual, expected)):
            with self.subTest(output=index):
                self.assertTrue(torch.equal(_bits(actual_i), _bits(expected_i)))


if __name__ == "__main__":
    unittest.main()
