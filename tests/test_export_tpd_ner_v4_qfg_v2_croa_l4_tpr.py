from __future__ import annotations

import copy
import unittest

import torch

from experiments.export_tpd_ner_v4_qfg_v2_croa_l4_tpr_to_inference import (
    EXPORT_SCHEMA,
    INFERENCE_PARAMETER_COUNT,
    INFERENCE_STATE_KEY_COUNT,
    TRAINING_STATE_KEY_COUNT,
    build_l4_tpr_inference_model_from_training_state_dict,
    export_l4_tpr_training_payload_to_inference,
    strip_l4_tpr_survival_state_dict,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    L4_TPR_STATE_KEYS,
    L4_TPR_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_l4_tpr_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    build_formal_v4_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


torch.set_num_threads(1)


class FormalL4TPRExporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training, _ = (
            build_formal_v4_qfg_v2_croa_l4_tpr_survival_model()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.training

    def test_exact_569_to_565_export_preserves_gate_and_qfg(self) -> None:
        training = copy.deepcopy(self.training)
        with torch.no_grad():
            training.ner_l4_tpr.reallocation_logits.copy_(
                torch.linspace(
                    -0.7,
                    0.9,
                    training.ner_l4_tpr.reallocation_logits.numel(),
                ).reshape_as(training.ner_l4_tpr.reallocation_logits)
            )
            for index, parameter in enumerate(
                training.target_survival.parameters(),
                start=1,
            ):
                parameter.fill_(0.01 * index)
        training_state = training.state_dict()
        self.assertEqual(len(training_state), TRAINING_STATE_KEY_COUNT)

        inference_state = strip_l4_tpr_survival_state_dict(training_state)
        self.assertEqual(len(inference_state), INFERENCE_STATE_KEY_COUNT)
        self.assertFalse(
            any(key.startswith(SURVIVAL_STATE_PREFIX) for key in inference_state)
        )
        self.assertEqual(
            {
                key
                for key in inference_state
                if key.startswith(L4_TPR_STATE_PREFIX)
            },
            set(L4_TPR_STATE_KEYS),
        )
        self.assertEqual(
            {key for key in inference_state if key.startswith(QFG_STATE_PREFIX)},
            set(QFG_STATE_KEYS),
        )
        for name in (*L4_TPR_STATE_KEYS, *QFG_STATE_KEYS):
            with self.subTest(preserved=name):
                self.assertTrue(
                    torch.equal(inference_state[name], training_state[name])
                )

        inference, metadata = (
            build_l4_tpr_inference_model_from_training_state_dict(
                training_state
            )
        )
        self.assertIs(
            type(inference),
            TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
        )
        self.assertFalse(hasattr(inference, "target_survival"))
        self.assertEqual(metadata["state_key_count"], INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in inference.parameters()),
            INFERENCE_PARAMETER_COUNT,
        )
        for name in L4_TPR_STATE_KEYS:
            self.assertTrue(
                torch.equal(inference.state_dict()[name], training_state[name])
            )

    def test_exported_graph_is_bitwise_identical_to_training_graph(self) -> None:
        training = copy.deepcopy(self.training)
        with torch.no_grad():
            training.ner_l4_tpr.reallocation_logits.fill_(0.31)
        inference, _ = build_l4_tpr_inference_model_from_training_state_dict(
            training.state_dict()
        )
        training.eval()
        inference.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080504)
        image = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_output = training(image)
            inference_output = inference(image)
        self.assertEqual(len(training_output), 6)
        self.assertEqual(len(inference_output), 6)
        for index, (actual, expected) in enumerate(
            zip(inference_output, training_output)
        ):
            with self.subTest(output=index):
                self.assertTrue(torch.equal(actual, expected))

    def test_payload_metadata_and_wrong_states_are_rejected(self) -> None:
        state = dict(self.training.state_dict())
        payload = export_l4_tpr_training_payload_to_inference(
            {
                "state_dict": state,
                "checkpoint_role": "best_miou",
                "checkpoint_identity": {"schema": "selected"},
                "run_identity": {"schema": "run"},
            }
        )
        self.assertEqual(payload["schema"], EXPORT_SCHEMA)
        self.assertEqual(
            set(payload["survival_state_removed"]), set(SURVIVAL_STATE_KEYS)
        )
        self.assertEqual(
            set(payload["l4_tpr_state_preserved"]), set(L4_TPR_STATE_KEYS)
        )
        self.assertEqual(
            set(payload["qfg_state_preserved"]), set(QFG_STATE_KEYS)
        )

        missing_tss = dict(state)
        missing_tss.pop(SURVIVAL_STATE_KEYS[0])
        with self.assertRaisesRegex(ValueError, "569 training state keys"):
            strip_l4_tpr_survival_state_dict(missing_tss)

        missing_gate = dict(state)
        missing_gate.pop(L4_TPR_STATE_KEYS[0])
        missing_gate["fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "one L4-TPR key"):
            strip_l4_tpr_survival_state_dict(missing_gate)

        foreign_tss = dict(state)
        foreign_tss["target_survival.fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "569 training state keys"):
            strip_l4_tpr_survival_state_dict(foreign_tss)

        current_final, _ = build_formal_v4_qfg_v2_croa_survival_model()
        self.assertEqual(len(current_final.state_dict()), 568)
        with self.assertRaisesRegex(ValueError, "569 training state keys"):
            strip_l4_tpr_survival_state_dict(current_final.state_dict())


if __name__ == "__main__":
    unittest.main()
