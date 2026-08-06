from __future__ import annotations

import copy
import unittest

import torch

from model.tpd_forward_contract import TPDForwardOutput
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2 import (
    FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_STATE_KEY_COUNT,
    PBDR_V2_STATE_KEYS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet,
    build_formal_v4_qfg_v2_croa_pbdr_v2_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v2_survival_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v2_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v2_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _six(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, TPDForwardOutput):
        value = value.segmentation
    if not isinstance(value, tuple) or len(value) != 6:
        raise TypeError("expected six segmentation outputs")
    return value


def _loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(
        float(index + 1) * output.square().mean()
        for index, output in enumerate(outputs)
    )


class FormalPBDRV2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.pbdr, cls.training_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v2_survival_model()
        )
        cls.inference, cls.inference_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v2_inference_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.pbdr
        del cls.inference
        del cls.training_metadata
        del cls.inference_metadata

    def test_builders_counts_manifest_and_shared_zero_extension(self) -> None:
        self.assertIs(
            type(self.pbdr),
            TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet,
        )
        self.assertIs(
            type(self.inference),
            TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet,
        )
        training = validate_formal_v4_qfg_v2_croa_pbdr_v2_survival_model(
            self.pbdr,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
            require_zero_initialized_pbdr_v2=True,
        )
        inference = validate_formal_v4_qfg_v2_croa_pbdr_v2_inference_model(
            self.inference,
            require_identity_initialized_qfg=True,
            require_zero_initialized_pbdr_v2=True,
        )
        self.assertEqual(
            training["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            inference["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertEqual(
            training["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            inference["total_parameters"],
            PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_PARAMETERS,
        )
        self.assertFalse(hasattr(self.inference, "target_survival"))
        self.assertEqual(
            self.training_metadata["architecture_manifest"][
                "deployment_graph"
            ],
            "v4_qfg_v2_croa_pbdr_v2_with_training_only_tss_heads",
        )
        self.assertEqual(
            self.inference_metadata["architecture_manifest"][
                "deployment_graph"
            ],
            "v4_qfg_v2_croa_pbdr_v2_no_tss",
        )

        current_state = self.current.state_dict()
        pbdr_state = self.pbdr.state_dict()
        self.assertEqual(
            set(current_state),
            set(pbdr_state) - set(PBDR_V2_STATE_KEYS),
        )
        for name, expected in current_state.items():
            self.assertTrue(torch.equal(_bits(pbdr_state[name]), _bits(expected)))
        for name in PBDR_V2_STATE_KEYS:
            self.assertEqual(int(torch.count_nonzero(pbdr_state[name])), 0)

    def test_zero_anchor_outputs_shared_gradients_and_first_adam(self) -> None:
        current = copy.deepcopy(self.current).eval()
        pbdr = copy.deepcopy(self.pbdr).eval()
        current.zero_grad(set_to_none=True)
        pbdr.zero_grad(set_to_none=True)
        generator = torch.Generator().manual_seed(2026080603)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        current_optimizer = torch.optim.Adam(current.parameters(), lr=1.0e-4)
        pbdr_optimizer = torch.optim.Adam(pbdr.parameters(), lr=1.0e-4)

        current_outputs = _six(current(image))
        pbdr_outputs = _six(pbdr(image))
        for actual, expected in zip(pbdr_outputs, current_outputs):
            self.assertTrue(torch.equal(_bits(actual), _bits(expected)))
        current_loss = _loss(current_outputs)
        pbdr_loss = _loss(pbdr_outputs)
        self.assertTrue(torch.equal(_bits(pbdr_loss), _bits(current_loss)))
        current_loss.backward()
        pbdr_loss.backward()

        current_parameters = dict(current.named_parameters())
        pbdr_parameters = dict(pbdr.named_parameters())
        shared_names = set(pbdr_parameters) - {
            name for name in pbdr_parameters if name.startswith("pbdr_v2.")
        }
        self.assertEqual(shared_names, set(current_parameters))
        for name in sorted(shared_names):
            expected = current_parameters[name].grad
            actual = pbdr_parameters[name].grad
            self.assertIs(actual is None, expected is None, msg=name)
            if expected is not None:
                self.assertTrue(
                    torch.equal(_bits(actual), _bits(expected)),
                    msg=name,
                )
        for name in (
            "pbdr_v2.direct_residual_projection.weight",
            "pbdr_v2.rescue_strength_raw",
            "pbdr_v2.suppression_strength_raw",
        ):
            gradient = pbdr_parameters[name].grad
            self.assertIsNotNone(gradient, msg=name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, msg=name)

        current_optimizer.step()
        pbdr_optimizer.step()
        for name in sorted(shared_names):
            self.assertTrue(
                torch.equal(
                    _bits(pbdr_parameters[name]),
                    _bits(current_parameters[name]),
                ),
                msg=name,
            )

    def test_training_to_inference_state_and_output_equivalence(self) -> None:
        training = copy.deepcopy(self.pbdr)
        with torch.no_grad():
            training.pbdr_v2.direct_residual_projection.weight.fill_(0.03)
            training.pbdr_v2.confidence_projection.weight.fill_(0.02)
            training.pbdr_v2.confidence_projection.bias.fill_(-0.1)
            training.pbdr_v2.rescue_strength_raw.fill_(0.2)
            training.pbdr_v2.suppression_strength_raw.fill_(0.3)
        training_state = training.state_dict()
        inference_state = {
            name: value
            for name, value in training_state.items()
            if name not in SURVIVAL_STATE_KEYS
        }
        self.assertEqual(
            len(inference_state),
            FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT,
        )
        inference = copy.deepcopy(self.inference)
        inference.load_state_dict(inference_state, strict=True)
        training.eval()
        inference.eval()
        generator = torch.Generator().manual_seed(2026080604)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_outputs = _six(training(image))
            inference_outputs = _six(inference(image))
        for actual, expected in zip(inference_outputs, training_outputs):
            self.assertTrue(torch.equal(actual, expected))

    def test_test_mode_returns_routed_single_probability(self) -> None:
        current = copy.deepcopy(self.current).eval()
        pbdr = copy.deepcopy(self.pbdr).eval()
        current.mode = "test"
        pbdr.mode = "test"
        image = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            current_value = current(image)
            zero_value = pbdr(image)
        self.assertTrue(torch.equal(current_value, zero_value))
        with torch.no_grad():
            pbdr.pbdr_v2.direct_residual_projection.weight.fill_(0.1)
            routed_value = pbdr(image)
        self.assertEqual(tuple(routed_value.shape), (1, 1, 32, 32))
        self.assertTrue(torch.isfinite(routed_value).all())
        self.assertFalse(torch.equal(routed_value, zero_value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_autocast_zero_anchor_keeps_dtype_and_values(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                current = copy.deepcopy(self.current).cuda().eval()
                pbdr = copy.deepcopy(self.pbdr).cuda().eval()
                current.zero_grad(set_to_none=True)
                pbdr.zero_grad(set_to_none=True)
                generator = torch.Generator(device="cuda").manual_seed(
                    2026080605
                )
                image = torch.randn(
                    1,
                    1,
                    32,
                    32,
                    generator=generator,
                    device="cuda",
                )
                with torch.autocast(device_type="cuda", dtype=dtype):
                    current_outputs = _six(current(image))
                    pbdr_outputs = _six(pbdr(image))
                    current_loss = _loss(current_outputs)
                    pbdr_loss = _loss(pbdr_outputs)
                for actual, expected in zip(pbdr_outputs, current_outputs):
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(
                        torch.equal(_bits(actual), _bits(expected))
                    )
                self.assertEqual(pbdr_loss.dtype, current_loss.dtype)
                self.assertTrue(
                    torch.equal(_bits(pbdr_loss), _bits(current_loss))
                )
                # The frozen formal1000 recipe is FP32, whose shared-gradient
                # equality is asserted above.  AMP is covered here only as a
                # forward identity/dtype regression; its backward reduction
                # order is not part of the formal training contract.
                del current, pbdr, image
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
