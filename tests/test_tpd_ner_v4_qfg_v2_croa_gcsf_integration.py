from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf import (
    FORMAL_V4_QFG_V2_CROA_GCSF_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_GCSF_SURVIVAL_STATE_KEY_COUNT,
    GCSF_STATE_KEYS,
    GCSF_STATE_PREFIX,
    PRODUCTION_V4_QFG_V2_CROA_GCSF_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_GCSF_SURVIVAL_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFSurvivalSCTransNet,
    build_formal_v4_qfg_v2_croa_gcsf_inference_model,
    build_formal_v4_qfg_v2_croa_gcsf_survival_model,
    validate_formal_v4_qfg_v2_croa_gcsf_inference_model,
    validate_formal_v4_qfg_v2_croa_gcsf_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _six_output_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(
        float(index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    )


class FormalGCSFIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.gcsf, cls.training_metadata = (
            build_formal_v4_qfg_v2_croa_gcsf_survival_model()
        )
        cls.inference, cls.inference_metadata = (
            build_formal_v4_qfg_v2_croa_gcsf_inference_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.gcsf
        del cls.inference
        del cls.training_metadata
        del cls.inference_metadata

    def test_formal_builders_counts_manifest_and_shared_scratch_init(self) -> None:
        training = validate_formal_v4_qfg_v2_croa_gcsf_survival_model(
            self.gcsf,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
            require_zero_initialized_gcsf=True,
        )
        inference = validate_formal_v4_qfg_v2_croa_gcsf_inference_model(
            self.inference,
            require_identity_initialized_qfg=True,
            require_zero_initialized_gcsf=True,
        )
        self.assertIs(
            type(self.gcsf),
            TPDNERV8MPRSDCHV4QFGV2CROAGCSFSurvivalSCTransNet,
        )
        self.assertIs(
            type(self.inference),
            TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
        )
        self.assertEqual(
            training["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_GCSF_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            training["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_GCSF_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            inference["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_GCSF_INFERENCE_PARAMETERS,
        )
        self.assertEqual(
            inference["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_GCSF_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertEqual(
            self.training_metadata["construction"],
            "scratch_seed42_no_parent_checkpoint",
        )
        self.assertFalse(hasattr(self.inference, "target_survival"))

        current_state = self.current.state_dict()
        gcsf_state = self.gcsf.state_dict()
        self.assertEqual(
            set(current_state),
            set(gcsf_state) - set(GCSF_STATE_KEYS),
        )
        for name, expected in current_state.items():
            with self.subTest(shared_scratch_state=name):
                self.assertTrue(torch.equal(_bits(gcsf_state[name]), _bits(expected)))
        for name in GCSF_STATE_KEYS:
            self.assertEqual(int(torch.count_nonzero(gcsf_state[name])), 0)

        self.assertIs(
            type(self.gcsf)._forward_with_relay,
            type(self.inference)._forward_with_relay,
        )

    def test_zero_anchor_full_model_output_shared_gradient_and_first_adam(self) -> None:
        current = copy.deepcopy(self.current).eval()
        gcsf = copy.deepcopy(self.gcsf).eval()
        current.zero_grad(set_to_none=True)
        gcsf.zero_grad(set_to_none=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080501)
        images = torch.randn(1, 1, 32, 32, generator=generator)

        current_optimizer = torch.optim.Adam(current.parameters(), lr=1.0e-4)
        gcsf_optimizer = torch.optim.Adam(gcsf.parameters(), lr=1.0e-4)
        current_outputs = current(images)
        with mock.patch.object(
            gcsf.global_skip_fusion,
            "forward",
            wraps=gcsf.global_skip_fusion.forward,
        ) as fusion_forward:
            gcsf_outputs = gcsf(images)
        self.assertEqual(fusion_forward.call_count, 1)
        self.assertEqual(len(current_outputs), 6)
        self.assertEqual(len(gcsf_outputs), 6)
        for index, (actual, expected) in enumerate(
            zip(gcsf_outputs, current_outputs)
        ):
            with self.subTest(output=index):
                self.assertTrue(torch.equal(_bits(actual), _bits(expected)))

        current_loss = _six_output_loss(current_outputs)
        gcsf_loss = _six_output_loss(gcsf_outputs)
        self.assertTrue(torch.equal(_bits(gcsf_loss), _bits(current_loss)))
        current_loss.backward()
        gcsf_loss.backward()

        current_parameters = dict(current.named_parameters())
        gcsf_parameters = dict(gcsf.named_parameters())
        shared_names = set(gcsf_parameters) - {
            name
            for name in gcsf_parameters
            if name.startswith(GCSF_STATE_PREFIX)
        }
        self.assertEqual(shared_names, set(current_parameters))
        for name in sorted(shared_names):
            expected_gradient = current_parameters[name].grad
            actual_gradient = gcsf_parameters[name].grad
            with self.subTest(shared_gradient=name):
                self.assertIs(
                    actual_gradient is None,
                    expected_gradient is None,
                )
                if expected_gradient is not None:
                    self.assertTrue(
                        torch.equal(_bits(actual_gradient), _bits(expected_gradient))
                    )

        gate_gradient_l1 = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in gcsf.global_skip_fusion.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gate_gradient_l1, 0.0)

        current_optimizer.step()
        gcsf_optimizer.step()
        expected_adam_keys = {"step", "exp_avg", "exp_avg_sq"}
        for name in sorted(shared_names):
            current_parameter = current_parameters[name]
            gcsf_parameter = gcsf_parameters[name]
            with self.subTest(first_adam_parameter=name):
                self.assertTrue(
                    torch.equal(_bits(gcsf_parameter), _bits(current_parameter))
                )
            current_state = current_optimizer.state.get(current_parameter)
            gcsf_state = gcsf_optimizer.state.get(gcsf_parameter)
            self.assertIs(gcsf_state is None, current_state is None, msg=name)
            if current_state is None:
                continue
            self.assertEqual(set(current_state), expected_adam_keys, msg=name)
            self.assertEqual(set(gcsf_state), expected_adam_keys, msg=name)
            for state_name in sorted(expected_adam_keys):
                with self.subTest(first_adam_named_state=f"{name}.{state_name}"):
                    self.assertTrue(
                        torch.equal(
                            _bits(gcsf_state[state_name]),
                            _bits(current_state[state_name]),
                        )
                    )

        changed_gcsf = 0
        for parameter in gcsf.global_skip_fusion.parameters():
            state = gcsf_optimizer.state.get(parameter)
            self.assertIsNotNone(state)
            self.assertEqual(set(state), expected_adam_keys)
            if int(torch.count_nonzero(parameter.detach())) > 0:
                changed_gcsf += 1
        self.assertGreater(changed_gcsf, 0)


if __name__ == "__main__":
    unittest.main()
