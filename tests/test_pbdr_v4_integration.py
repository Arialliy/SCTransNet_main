from __future__ import annotations

import inspect
import unittest

import torch
import torch.nn as nn

from experiments.pbdr_v4_state_contract import PBDRV4StateContractError
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4 import (
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT,
    PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
    PBDR_V4_STATE_KEYS,
    PBDRV4IntegrationError,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet,
    build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint,
    build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model,
    build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model,
    strip_training_only_survival_state,
    validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model,
    validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model,
    warm_start_formal_pbdr_v4_from_current,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
)


torch.set_num_threads(1)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


class FormalPBDRV4IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.current_state = _clone_state(cls.current)
        cls.miou, cls.miou_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
                role="best_miou"
            )
        )
        cls.pd, cls.pd_metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
                role="best_pd"
            )
        )
        cls.miou_warm_start = warm_start_formal_pbdr_v4_from_current(
            cls.miou,
            cls.current_state,
        )
        cls.pd_warm_start = warm_start_formal_pbdr_v4_from_current(
            cls.pd,
            cls.current_state,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.current_state
        del cls.miou
        del cls.pd
        del cls.miou_metadata
        del cls.pd_metadata
        del cls.miou_warm_start
        del cls.pd_warm_start

    def test_builders_are_role_explicit_and_counts_include_v4_state(self) -> None:
        survival_signature = inspect.signature(
            build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model
        )
        inference_signature = inspect.signature(
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model
        )
        self.assertIs(
            survival_signature.parameters["role"].default,
            inspect.Parameter.empty,
        )
        self.assertIs(
            inference_signature.parameters["role"].default,
            inspect.Parameter.empty,
        )

        for model, role, metadata in (
            (self.miou, "best_miou", self.miou_metadata),
            (self.pd, "best_pd", self.pd_metadata),
        ):
            self.assertIs(
                type(model),
                TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet,
            )
            self.assertEqual(model.pbdr_v4_role, role)
            self.assertEqual(model.pbdr_v4.role, role)
            self.assertEqual(
                len(model.state_dict()),
                FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT,
            )
            self.assertEqual(
                _parameter_count(model),
                PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_PARAMETERS,
            )
            self.assertEqual(
                tuple(
                    name
                    for name in model.state_dict()
                    if name.startswith("pbdr_v4.")
                ),
                PBDR_V4_STATE_KEYS,
            )
            manifest = model.architecture_manifest()
            self.assertEqual(manifest["pbdr_v4_role"], role)
            self.assertEqual(
                manifest["pbdr_v4_auxiliary_logit_order"],
                ("gt2", "gt3", "gt4", "gt5"),
            )
            self.assertEqual(manifest["pbdr_v4_state_key_count"], 27)
            self.assertEqual(
                manifest["pbdr_v4_core_manifest"]["persistent_buffer_names"],
                ("role_code", "positive_limit", "negative_limit"),
            )
            self.assertEqual(metadata["role"], role)

        self.assertEqual(
            self.miou_warm_start["expected_missing_extension_key_count"],
            27,
        )
        self.assertEqual(
            self.miou_warm_start["expected_missing_extension_keys"],
            PBDR_V4_STATE_KEYS,
        )
        self.assertTrue(
            self.miou_warm_start["all_current_tensors_bitwise_equal_after_load"]
        )

    def test_cross_role_full_model_state_load_is_rejected(self) -> None:
        target, _ = build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
            role="best_miou"
        )
        with self.assertRaisesRegex(RuntimeError, "role_code"):
            target.load_state_dict(self.pd.state_dict(), strict=False)

    def test_single_forward_identity_shapes_and_named_auxiliary_order(self) -> None:
        current = self.current.eval()
        candidate = self.miou.eval()
        current.mode = "train"
        candidate.mode = "train"
        generator = torch.Generator().manual_seed(2026080711)
        image = torch.randn(1, 1, 32, 32, generator=generator)

        with torch.no_grad():
            current_probabilities = current(image)
            probabilities, auxiliary = candidate.forward_for_pbdr_v4_training(
                image
            )
        self.assertIsInstance(current_probabilities, tuple)
        self.assertEqual(len(current_probabilities), 6)
        self.assertEqual(len(probabilities), 6)
        for actual, expected in zip(probabilities, current_probabilities):
            self.assertEqual(tuple(actual.shape), (1, 1, 32, 32))
            self.assertTrue(torch.equal(actual, expected))

        ordered = auxiliary.ordered_deep_supervision_logits()
        self.assertIs(ordered[0], auxiliary.gt2_logits)
        self.assertIs(ordered[1], auxiliary.gt3_logits)
        self.assertIs(ordered[2], auxiliary.gt4_logits)
        self.assertIs(ordered[3], auxiliary.gt5_logits)
        self.assertTrue(
            torch.equal(probabilities[0], torch.sigmoid(auxiliary.gt5_logits))
        )
        self.assertTrue(
            torch.equal(probabilities[1], torch.sigmoid(auxiliary.gt4_logits))
        )
        self.assertTrue(
            torch.equal(probabilities[2], torch.sigmoid(auxiliary.gt3_logits))
        )
        self.assertTrue(
            torch.equal(probabilities[3], torch.sigmoid(auxiliary.gt2_logits))
        )
        self.assertTrue(
            torch.equal(probabilities[4], torch.sigmoid(auxiliary.d0_logits))
        )
        self.assertTrue(
            torch.equal(
                auxiliary.candidate_base_logits,
                auxiliary.routed_logits,
            )
        )
        self.assertEqual(int(torch.count_nonzero(auxiliary.delta_logits)), 0)
        self.assertIs(auxiliary.delta_logits, auxiliary.routing.delta_logits)
        self.assertIs(auxiliary.routed_logits, auxiliary.routing.routed_logits)

        with torch.no_grad():
            ordinary_training = candidate(image)
        self.assertIsInstance(ordinary_training, tuple)
        self.assertEqual(len(ordinary_training), 6)
        candidate.mode = "test"
        with torch.no_grad():
            ordinary_test = candidate(image)
        self.assertIsInstance(ordinary_test, torch.Tensor)
        self.assertEqual(tuple(ordinary_test.shape), (1, 1, 32, 32))
        self.assertTrue(torch.equal(ordinary_test, ordinary_training[-1]))
        candidate.mode = "train"

    def test_training_to_inference_state_and_complete_payload_export(self) -> None:
        training_state = _clone_state(self.miou)
        stripped = strip_training_only_survival_state(training_state)
        self.assertEqual(
            len(stripped),
            FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertFalse(set(SURVIVAL_STATE_KEYS) & set(stripped))
        self.assertTrue(set(PBDR_V4_STATE_KEYS) <= set(stripped))

        inference, raw = build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
            role="best_miou"
        )
        incompatible = inference.load_state_dict(stripped, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
            inference,
            expected_role="best_miou",
        )
        self.assertIs(
            type(inference),
            TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet,
        )
        self.assertEqual(
            validation["state_key_count"],
            FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertEqual(
            _parameter_count(inference),
            PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_PARAMETERS,
        )
        self.assertTrue(raw["warm_start_required"])

        payload = {
            "schema": PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
            "role": "best_miou",
            "stage": "stage1",
            "architecture_manifest": self.miou.architecture_manifest(),
            "state_dict": training_state,
        }
        exported, metadata = (
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
                payload,
                expected_role="best_miou",
                expected_stage="stage1",
                current_state=self.current_state,
            )
        )
        self.assertFalse(exported.training)
        self.assertEqual(exported.mode, "test")
        self.assertEqual(metadata["role"], "best_miou")
        self.assertEqual(metadata["stage"], "stage1")
        self.assertTrue(metadata["strict_checkpoint_payload"])
        self.assertTrue(
            metadata["training_validation"]["state_contract"][
                "all_base_buffers_bitwise_current"
            ]
        )

        image = torch.randn(
            1,
            1,
            32,
            32,
            generator=torch.Generator().manual_seed(2026080712),
        )
        training = self.miou.eval()
        training.mode = "test"
        with torch.no_grad():
            training_output = training(image)
            exported_output = exported(image)
        self.assertTrue(torch.equal(training_output, exported_output))
        training.mode = "train"

        with self.assertRaisesRegex(PBDRV4IntegrationError, "incomplete"):
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
                training_state,
                expected_role="best_miou",
                expected_stage="stage1",
                current_state=self.current_state,
            )
        bad_schema = dict(payload)
        bad_schema["schema"] = "wrong"
        with self.assertRaisesRegex(PBDRV4IntegrationError, "schema"):
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
                bad_schema,
                expected_role="best_miou",
                expected_stage="stage1",
                current_state=self.current_state,
            )
        bad_role = dict(payload)
        bad_role["role"] = "best_pd"
        with self.assertRaisesRegex(PBDRV4IntegrationError, "role"):
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
                bad_role,
                expected_role="best_miou",
                expected_stage="stage1",
                current_state=self.current_state,
            )
        bad_stage = dict(payload)
        bad_stage["stage"] = "stage2"
        with self.assertRaisesRegex(PBDRV4IntegrationError, "stage"):
            build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
                bad_stage,
                expected_role="best_miou",
                expected_stage="stage1",
                current_state=self.current_state,
            )

    def test_stage2_allows_only_named_parameters_and_rejects_bn_buffer(self) -> None:
        candidate, _ = build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
            role="best_pd"
        )
        warm_start_formal_pbdr_v4_from_current(candidate, self.current_state)
        with torch.no_grad():
            candidate.outc.weight.add_(1.0e-3)
            first_up_parameter = next(candidate.up_decoder1.parameters())
            first_up_parameter.add_(1.0e-4)

        stage2 = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
            candidate,
            expected_role="best_pd",
            current_state=self.current_state,
            stage="stage2",
        )
        permitted = stage2["state_contract"][
            "permitted_changed_parameter_names"
        ]
        self.assertTrue(any(name.startswith("outc.") for name in permitted))
        self.assertTrue(
            any(name.startswith("up_decoder1.") for name in permitted)
        )
        with self.assertRaisesRegex(
            PBDRV4StateContractError,
            "immutable parameter",
        ):
            validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
                candidate,
                expected_role="best_pd",
                current_state=self.current_state,
                stage="stage1",
            )

        bn_name, bn_buffer = next(
            (name, value)
            for name, value in candidate.named_buffers()
            if name.startswith("up_decoder1.") and name.endswith("running_mean")
        )
        with torch.no_grad():
            bn_buffer.add_(1.0)
        with self.assertRaisesRegex(
            PBDRV4StateContractError,
            "immutable buffer",
        ):
            validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
                candidate,
                expected_role="best_pd",
                current_state=self.current_state,
                stage="stage2",
            )
        self.assertTrue(bn_name.startswith("up_decoder1."))


if __name__ == "__main__":
    unittest.main()
