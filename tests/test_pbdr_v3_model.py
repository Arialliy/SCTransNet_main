from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from model.tpd_conservative_residual_calibrator_v3 import (
    ConservativeResidualCalibratorV3,
    FORMAL_RESIDUAL_LIMIT,
    PBDR_V3_LOCAL_STATE_KEYS,
    PRODUCTION_PBDR_V3_PARAMETERS,
    validate_formal_pbdr_v3_calibrator,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3 import (
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT,
    PBDR_V3_STATE_KEYS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet,
    _install_formal_pbdr_v3,
    build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class ConservativeResidualCalibratorV3Tests(unittest.TestCase):
    def test_exact_identity_bounded_gates_and_first_step_gradients(self) -> None:
        generator = torch.Generator().manual_seed(2026080601)
        calibrator = ConservativeResidualCalibratorV3()
        z_out = torch.randn(
            2,
            1,
            8,
            8,
            generator=generator,
            requires_grad=True,
        )
        z_d0 = torch.randn(
            2,
            1,
            8,
            8,
            generator=generator,
            requires_grad=True,
        )
        q4 = torch.randn(
            2,
            8,
            3,
            3,
            generator=generator,
            requires_grad=True,
        )
        local = torch.randn(
            2,
            32,
            8,
            8,
            generator=generator,
            requires_grad=True,
        )
        routing = calibrator.forward_with_diagnostics(
            z_out,
            z_d0,
            q4,
            local,
        )
        self.assertTrue(torch.equal(routing.routed_logits, z_out))
        self.assertEqual(int(torch.count_nonzero(routing.delta_logits)), 0)
        self.assertTrue(torch.equal(routing.rescue_gate, routing.suppression_gate))
        self.assertTrue(bool((routing.rescue_gate >= 0.0).all()))
        self.assertTrue(bool((routing.rescue_gate <= 1.0).all()))

        weight = torch.linspace(
            -0.75,
            1.25,
            routing.routed_logits.numel(),
        ).reshape_as(routing.routed_logits)
        loss = (routing.routed_logits * weight).sum()
        loss.backward()
        self.assertTrue(torch.equal(z_out.grad, weight))
        self.assertIsNone(z_d0.grad)
        self.assertIsNone(q4.grad)
        self.assertIsNone(local.grad)
        terminal = calibrator.routing_trunk[-1]
        self.assertIsNotNone(terminal.weight.grad)
        self.assertIsNotNone(terminal.bias.grad)
        self.assertGreater(float(terminal.weight.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(terminal.weight.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(terminal.bias.grad[0].abs()), 0.0)
        self.assertGreater(float(terminal.bias.grad[1].abs()), 0.0)

        calibrator.zero_grad(set_to_none=True)
        with torch.no_grad():
            terminal.weight.zero_()
            terminal.bias.copy_(torch.tensor((20.0, -20.0)))
        shifted = calibrator.forward_with_diagnostics(
            z_out.detach(),
            z_d0.detach(),
            q4.detach(),
            local.detach(),
        )
        self.assertGreater(float(shifted.delta_logits.detach().max()), 0.0)
        self.assertLessEqual(
            float(shifted.delta_logits.detach().abs().max()),
            FORMAL_RESIDUAL_LIMIT,
        )

    def test_safe_q4_does_not_amplify_weak_centered_evidence(self) -> None:
        calibrator = ConservativeResidualCalibratorV3(evidence_floor=1.0)
        generator = torch.Generator().manual_seed(2026080602)
        q4 = 3.0 + 1.0e-4 * torch.randn(
            2,
            8,
            5,
            7,
            generator=generator,
        )
        centered = q4 - q4.mean(dim=(2, 3), keepdim=True)
        normalized = calibrator._safe_q4(q4)
        self.assertTrue(torch.equal(normalized, centered))
        self.assertFalse(normalized.requires_grad)
        self.assertLess(float(normalized.abs().max()), 1.0e-3)

    def test_formal_state_contract(self) -> None:
        calibrator = ConservativeResidualCalibratorV3()
        manifest = validate_formal_pbdr_v3_calibrator(
            calibrator,
            require_identity_initialization=True,
        )
        self.assertEqual(_parameter_count(calibrator), PRODUCTION_PBDR_V3_PARAMETERS)
        self.assertEqual(tuple(calibrator.state_dict()), PBDR_V3_LOCAL_STATE_KEYS)
        self.assertFalse(manifest["direct_q4_residual"])
        self.assertFalse(manifest["direct_d0_residual"])
        self.assertTrue(manifest["terminal_gate_first_derivative_nonzero"])


class _DummyCurrent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.outc = nn.Conv2d(32, 1, kernel_size=1)


class FormalPBDRV3IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.pbdr, cls.training_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model()
        )
        cls.inference, cls.inference_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.pbdr
        del cls.inference
        del cls.training_metadata
        del cls.inference_metadata

    def test_install_is_deterministic_and_restores_ambient_rng(self) -> None:
        first = _DummyCurrent()
        torch.manual_seed(731)
        state_before = torch.random.get_rng_state().clone()
        _install_formal_pbdr_v3(first)
        state_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(state_after, state_before))

        second = _DummyCurrent()
        torch.manual_seed(997)
        second_state_before = torch.random.get_rng_state().clone()
        _install_formal_pbdr_v3(second)
        self.assertTrue(
            torch.equal(torch.random.get_rng_state(), second_state_before)
        )
        for name, expected in first.pbdr_v3.state_dict().items():
            self.assertTrue(
                torch.equal(second.pbdr_v3.state_dict()[name], expected),
                msg=name,
            )

    def test_builders_counts_manifest_and_current_warm_start(self) -> None:
        self.assertIs(
            type(self.pbdr),
            TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet,
        )
        self.assertIs(
            type(self.inference),
            TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet,
        )
        training = validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
            self.pbdr,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
            require_identity_initialized_pbdr_v3=True,
        )
        inference = validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(
            self.inference,
            require_identity_initialized_qfg=True,
            require_identity_initialized_pbdr_v3=True,
        )
        self.assertEqual(
            training["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            inference["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertEqual(
            training["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            inference["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_PARAMETERS,
        )
        self.assertTrue(self.training_metadata["warm_start_required"])
        self.assertTrue(self.inference_metadata["warm_start_required"])
        self.assertFalse(hasattr(self.inference, "target_survival"))

        candidate = copy.deepcopy(self.pbdr)
        incompatible = candidate.load_state_dict(
            self.current.state_dict(),
            strict=False,
        )
        self.assertEqual(tuple(incompatible.missing_keys), PBDR_V3_STATE_KEYS)
        self.assertEqual(incompatible.unexpected_keys, [])
        current_state = self.current.state_dict()
        candidate_state = candidate.state_dict()
        self.assertEqual(
            set(current_state),
            set(candidate_state) - set(PBDR_V3_STATE_KEYS),
        )
        for name, expected in current_state.items():
            self.assertTrue(torch.equal(candidate_state[name], expected), msg=name)

    def test_identity_forward_explicit_aux_and_stage1_gradients(self) -> None:
        current = copy.deepcopy(self.current).eval()
        candidate = copy.deepcopy(self.pbdr).eval()
        incompatible = candidate.load_state_dict(
            current.state_dict(),
            strict=False,
        )
        self.assertEqual(tuple(incompatible.missing_keys), PBDR_V3_STATE_KEYS)
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        for parameter in candidate.pbdr_v3.parameters():
            parameter.requires_grad_(True)
        candidate.pbdr_v3.train()

        generator = torch.Generator().manual_seed(2026080603)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            current_probabilities = current(image)
            candidate_probabilities = candidate(image)
        self.assertIsInstance(current_probabilities, tuple)
        self.assertIsInstance(candidate_probabilities, tuple)
        self.assertEqual(len(current_probabilities), 6)
        self.assertEqual(len(candidate_probabilities), 6)
        for actual, expected in zip(
            candidate_probabilities,
            current_probabilities,
        ):
            self.assertTrue(torch.equal(actual, expected))

        probabilities, auxiliary = (
            candidate.forward_for_pbdr_v3_training(image)
        )
        self.assertEqual(len(probabilities), 6)
        self.assertEqual(len(auxiliary.auxiliary_logits), 5)
        self.assertTrue(
            torch.equal(probabilities[-1], torch.sigmoid(auxiliary.routed_logits))
        )
        self.assertTrue(
            torch.equal(auxiliary.base_logits, auxiliary.routed_logits)
        )
        self.assertEqual(
            int(torch.count_nonzero(auxiliary.routing.delta_logits)),
            0,
        )

        spatial_weight = torch.linspace(
            -0.5,
            1.5,
            auxiliary.routed_logits.numel(),
        ).reshape_as(auxiliary.routed_logits)
        loss = (auxiliary.routed_logits * spatial_weight).sum()
        loss.backward()
        parameters = dict(candidate.named_parameters())
        terminal_weight = parameters["pbdr_v3.routing_trunk.2.weight"]
        terminal_bias = parameters["pbdr_v3.routing_trunk.2.bias"]
        self.assertGreater(float(terminal_weight.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(terminal_weight.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(terminal_bias.grad[0].abs()), 0.0)
        self.assertGreater(float(terminal_bias.grad[1].abs()), 0.0)
        for name, parameter in parameters.items():
            if not name.startswith("pbdr_v3."):
                self.assertIsNone(parameter.grad, msg=name)

    def test_survival_to_inference_state_and_routed_output_equivalence(self) -> None:
        training = copy.deepcopy(self.pbdr)
        terminal = training.pbdr_v3.routing_trunk[-1]
        with torch.no_grad():
            terminal.weight.zero_()
            terminal.bias.copy_(torch.tensor((-1.0, -3.0)))
        inference_state = {
            name: value
            for name, value in training.state_dict().items()
            if name not in SURVIVAL_STATE_KEYS
        }
        self.assertEqual(
            len(inference_state),
            FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT,
        )
        inference = copy.deepcopy(self.inference)
        inference.load_state_dict(inference_state, strict=True)
        training.eval()
        inference.eval()
        training.mode = "test"
        inference.mode = "test"
        generator = torch.Generator().manual_seed(2026080604)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_output = training(image)
            inference_output = inference(image)
        self.assertEqual(tuple(training_output.shape), (1, 1, 32, 32))
        self.assertTrue(torch.equal(inference_output, training_output))
        self.assertTrue(torch.isfinite(inference_output).all())


if __name__ == "__main__":
    unittest.main()
