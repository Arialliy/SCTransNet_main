from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from experiments.tpd_training_loss import compute_tpd_training_loss
from model.tpd_forward_contract import TPDForwardOutput, evaluator_prediction
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    validate_formal_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (
    PersistentEvidencePositiveRoutingRelay,
)
from model.tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival import (
    FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS,
    PRODUCTION_V5_PER_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    build_formal_v5_per_qfg_v2_croa_inference_model,
    build_formal_v5_per_qfg_v2_croa_survival_model,
    validate_formal_v5_per_qfg_v2_croa_inference_model,
    validate_formal_v5_per_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260804)
    images = torch.randn(2, 1, 32, 32, generator=generator)
    target = torch.zeros(2, 1, 32, 32)
    target[0, 0, 3:6, 5:8] = 1.0
    target[1, 0, 20:23, 24:27] = 1.0
    return images, target


class NERV5PERIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training_model, cls.training_metadata = (
            build_formal_v5_per_qfg_v2_croa_survival_model()
        )
        cls.inference_model, cls.inference_metadata = (
            build_formal_v5_per_qfg_v2_croa_inference_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.training_model
        del cls.inference_model
        del cls.training_metadata
        del cls.inference_metadata

    def setUp(self) -> None:
        self.training_model.train()
        self.training_model.zero_grad(set_to_none=True)
        self.inference_model.train()
        self.inference_model.zero_grad(set_to_none=True)

    def test_formal_builders_validators_state_and_manifest(self) -> None:
        training = validate_formal_v5_per_qfg_v2_croa_survival_model(
            self.training_model,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
        )
        inference = validate_formal_v5_per_qfg_v2_croa_inference_model(
            self.inference_model,
            require_identity_initialized_qfg=True,
        )
        self.assertIs(
            type(self.training_model.tpd_ner),
            PersistentEvidencePositiveRoutingRelay,
        )
        self.assertIs(
            type(self.inference_model.tpd_ner),
            PersistentEvidencePositiveRoutingRelay,
        )
        self.assertEqual(
            training["state_key_count"],
            FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        )
        self.assertEqual(
            inference["state_key_count"],
            FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertEqual(
            training["total_parameters"],
            PRODUCTION_V5_PER_QFG_V2_CROA_SURVIVAL_PARAMETERS,
        )
        self.assertEqual(
            inference["total_parameters"],
            PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS,
        )
        self.assertFalse(
            training["checkpoint_semantically_interchangeable_with_v4"]
        )
        self.assertFalse(hasattr(self.inference_model, "target_survival"))
        self.assertFalse(
            any(
                key.startswith(SURVIVAL_STATE_PREFIX)
                for key in self.inference_model.state_dict()
            )
        )

        with self.assertRaisesRegex(TypeError, "exact integration class"):
            validate_formal_qfg_v2_croa_survival_model(self.training_model)

    def test_tss_off_loss_does_not_consume_training_only_heads(self) -> None:
        model = copy.deepcopy(self.training_model).train()
        images, target = _fixed_batch()
        output = model(images)
        self.assertIsInstance(output, TPDForwardOutput)
        criterion = nn.BCELoss(reduction="mean")
        losses = compute_tpd_training_loss(
            output,
            target,
            criterion,
            survival_weight=0.0,
            survival_pos_weight=1.0,
        )
        manual = sum(
            criterion(probability, target)
            for probability in output.segmentation
        )
        self.assertTrue(torch.equal(losses.total, manual))
        self.assertTrue(torch.equal(losses.segmentation, manual))
        self.assertEqual(losses.survival_terms, ())
        losses.total.backward()
        for parameter in model.target_survival.parameters():
            self.assertIsNone(parameter.grad)

        shared_prefixes = (
            "mtc.embeddings_1.",
            "tpd_ner.",
            "tpd_qfg.",
            "up_decoder2.",
        )
        shared = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith(shared_prefixes)
        ]
        self.assertTrue(shared)
        self.assertTrue(
            all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in shared
            )
        )

    def test_head_free_inference_matches_training_eval_segmentation(self) -> None:
        training_state = self.training_model.state_dict()
        inference_state = {
            key: value
            for key, value in training_state.items()
            if not key.startswith(SURVIVAL_STATE_PREFIX)
        }
        incompatible = self.inference_model.load_state_dict(
            inference_state,
            strict=True,
        )
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)
        self.training_model.eval()
        self.inference_model.eval()
        images, _ = _fixed_batch()
        with torch.no_grad():
            training_output = self.training_model(images[:1])
            inference_output = self.inference_model(images[:1])
        self.assertIsInstance(training_output, tuple)
        self.assertIsInstance(inference_output, tuple)
        self.assertEqual(len(training_output), 6)
        self.assertEqual(len(inference_output), 6)
        torch.testing.assert_close(
            evaluator_prediction(training_output),
            evaluator_prediction(inference_output),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
