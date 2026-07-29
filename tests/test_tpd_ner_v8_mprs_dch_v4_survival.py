from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from experiments.train_tpd_clean_v8_mprs_dch import (
    build_clean_v8_mprs_dch_model,
)
from model.tpd_forward_contract import TPDForwardOutput, evaluator_prediction
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
    TPDNERV8MPRSDCHV4SCTransNet,
    TailDCSupportMode,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_SURVIVAL_VARIANT,
    FORMAL_V4_PARENT_STATE_KEY_COUNT,
    FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_SURVIVAL_PARAMETERS,
    PRODUCTION_V4_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    TPDNERV8MPRSDCHV4SurvivalSCTransNet,
    build_formal_v4_survival_model,
    validate_formal_survival_model,
)


torch.set_num_threads(1)


class V4SurvivalModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = build_formal_v4_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        del cls.metadata

    def setUp(self) -> None:
        self.model.train()
        self.model.zero_grad(set_to_none=True)

    def test_constructor_keeps_v4_signature_and_rejects_completed_parent(
        self,
    ) -> None:
        parent_signature = inspect.signature(
            TPDNERV8MPRSDCHV4SCTransNet.__init__
        )
        extension_signature = inspect.signature(
            TPDNERV8MPRSDCHV4SurvivalSCTransNet.__init__
        )
        self.assertEqual(
            tuple(parent_signature.parameters),
            tuple(extension_signature.parameters),
        )
        for name, parent_parameter in parent_signature.parameters.items():
            extension_parameter = extension_signature.parameters[name]
            self.assertEqual(
                parent_parameter.kind,
                extension_parameter.kind,
                msg=name,
            )
            self.assertEqual(
                parent_parameter.default,
                extension_parameter.default,
                msg=name,
            )

        with self.assertRaisesRegex(ValueError, "raw Clean-V8 parent"):
            TPDNERV8MPRSDCHV4SurvivalSCTransNet(
                self.model,
                variant=FORMAL_SURVIVAL_VARIANT,
            )

    def test_formal_state_parameter_manifest_and_zero_contract(self) -> None:
        state = self.model.state_dict()
        survival_keys = {
            key for key in state if key.startswith("target_survival.")
        }
        self.assertEqual(len(state), FORMAL_V4_SURVIVAL_STATE_KEY_COUNT)
        self.assertEqual(
            len(state) - len(survival_keys),
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertEqual(survival_keys, set(SURVIVAL_STATE_KEYS))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            PRODUCTION_V4_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in self.model.target_survival.parameters()
            ),
            PRODUCTION_SURVIVAL_PARAMETERS,
        )
        self.assertFalse(tuple(self.model.target_survival.named_buffers()))
        for key in SURVIVAL_STATE_KEYS:
            self.assertEqual(int(torch.count_nonzero(state[key])), 0, msg=key)

        validated = validate_formal_survival_model(
            self.model,
            require_zero_initialized_heads=True,
        )
        self.assertEqual(
            validated["state_key_count"],
            FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            validated["parent_state_key_count"],
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertEqual(
            validated["total_parameters"],
            PRODUCTION_V4_SURVIVAL_PARAMETERS,
        )
        manifest = validated["architecture_manifest"]
        self.assertEqual(manifest["survival_endpoints"], ("emb1", "emb2"))
        self.assertEqual(manifest["survival_endpoint_grid"], "stride_16")
        self.assertEqual(manifest["survival_parameters"], 98)
        self.assertTrue(manifest["survival_training_only"])
        self.assertFalse(manifest["segmentation_path_modified"])
        self.assertFalse(manifest["inference_heads_required"])

        reference = next(
            parameter
            for name, parameter in self.model.named_parameters()
            if not name.startswith("target_survival.")
        )
        for parameter in self.model.target_survival.parameters():
            self.assertEqual(parameter.device, reference.device)
            self.assertEqual(parameter.dtype, reference.dtype)

    def test_formal_validator_checks_non_state_architecture(self) -> None:
        original = self.model.tpd_ner._dc_support_mode
        self.model.tpd_ner._dc_support_mode = TailDCSupportMode.LEGACY_GLOBAL
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "complement-tail",
            ):
                validate_formal_survival_model(self.model)
        finally:
            self.model.tpd_ner._dc_support_mode = original

    def test_head_construction_is_rng_neutral(self) -> None:
        parent, _ = build_clean_v8_mprs_dch_model(
            FORMAL_SURVIVAL_VARIANT,
            seed=42,
        )
        before = torch.get_rng_state().clone()
        extension = TPDNERV8MPRSDCHV4SurvivalSCTransNet(
            parent,
            variant=FORMAL_SURVIVAL_VARIANT,
            relay_width=DEFAULT_RELAY_WIDTH,
            relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
            dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
            tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
        )
        after = torch.get_rng_state()
        self.assertTrue(torch.equal(before, after))
        del parent, extension

    def test_training_capture_is_single_pass_and_eval_is_legacy(self) -> None:
        images = torch.randn(2, 1, 32, 32)
        forward1 = self.model.mtc.embeddings_1.forward_with_evidence
        forward2 = self.model.mtc.embeddings_2.forward_with_evidence
        with (
            mock.patch.object(
                self.model.mtc.embeddings_1,
                "forward_with_evidence",
                wraps=forward1,
            ) as capture1,
            mock.patch.object(
                self.model.mtc.embeddings_2,
                "forward_with_evidence",
                wraps=forward2,
            ) as capture2,
        ):
            output = self.model(images)

        self.assertIsInstance(output, TPDForwardOutput)
        self.assertEqual(capture1.call_count, 1)
        self.assertEqual(capture2.call_count, 1)
        self.assertIsInstance(output.segmentation, tuple)
        self.assertEqual(len(output.segmentation), 6)
        self.assertEqual(tuple(output.emb1_endpoint.shape), (2, 32, 2, 2))
        self.assertEqual(tuple(output.emb2_endpoint.shape), (2, 64, 2, 2))
        self.assertEqual(
            tuple(output.emb1_survival_logits.shape),
            (2, 1, 2, 2),
        )
        self.assertEqual(
            tuple(output.emb2_survival_logits.shape),
            (2, 1, 2, 2),
        )
        self.assertFalse(self.model._survival_capture_active)
        self.assertIsNone(self.model._captured_survival_endpoints)

        self.model.eval()
        with (
            torch.no_grad(),
            mock.patch.object(
                self.model.target_survival,
                "forward",
                wraps=self.model.target_survival.forward,
            ) as survival_forward,
        ):
            legacy = self.model(images[:1])
        self.assertIsInstance(legacy, tuple)
        self.assertEqual(len(legacy), 6)
        self.assertEqual(survival_forward.call_count, 0)
        self.assertIs(evaluator_prediction(legacy), legacy[-1])


if __name__ == "__main__":
    unittest.main()
