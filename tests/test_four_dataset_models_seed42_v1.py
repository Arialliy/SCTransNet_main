from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from experiments import four_dataset_models_seed42_v1 as builder
from model.SCTransNet import SCTransNet
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


torch.set_num_threads(1)


class FourDatasetModelsSeed42Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original, cls.final, cls.metadata = (
            builder.build_paired_paper_models("SIRST3")
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.original
        del cls.final

    def test_true_scratch_pair_has_bitwise_equal_shared_state(self) -> None:
        self.assertIs(type(self.original), SCTransNet)
        self.assertIs(
            type(self.final),
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
        )
        self.assertEqual(
            self.metadata["training_seed"],
            builder.TRAINING_SEED,
        )
        self.assertEqual(
            self.metadata["allowed_training_seeds"],
            [builder.TRAINING_SEED],
        )
        self.assertEqual(
            self.metadata["initialization_mode"],
            "true_scratch",
        )
        self.assertIsNone(self.metadata["parent_checkpoint"])
        self.assertEqual(self.metadata["parent_checkpoint_load_count"], 0)
        self.assertFalse(self.metadata["warm_start_used"])
        self.assertEqual(self.metadata["shared_state_key_count"], 504)
        self.assertEqual(self.metadata["original_only_state_key_count"], 6)
        self.assertEqual(self.metadata["final_only_state_key_count"], 64)
        self.assertTrue(self.metadata["shared_state_bitwise_equal"])

        original_state = self.original.state_dict()
        final_state = self.final.state_dict()
        for key in self.metadata["shared_state_keys"]:
            self.assertEqual(original_state[key].shape, final_state[key].shape)
            self.assertEqual(original_state[key].dtype, final_state[key].dtype)
            self.assertTrue(torch.equal(original_state[key], final_state[key]))

    def test_final_only_initialization_is_stable_and_zero_contracts_hold(
        self,
    ) -> None:
        torch.manual_seed(builder.TRAINING_SEED)
        rng_before = torch.get_rng_state().clone()
        _, repeated_final, repeated = builder.build_paired_models()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(
            repeated["derived_initialization_seeds"],
            self.metadata["derived_initialization_seeds"],
        )
        self.assertEqual(
            repeated["final_only_state_sha256"],
            self.metadata["final_only_state_sha256"],
        )
        self.assertEqual(
            repeated["final"]["state_sha256"],
            self.metadata["final"]["state_sha256"],
        )
        self.assertTrue(self.metadata["final"]["tss_zero_initialized"])
        self.assertTrue(
            self.metadata["final"]["qfg_terminal_zero_initialized"]
        )
        derived = self.metadata["derived_initialization_seeds"]
        self.assertEqual(set(derived), {"tpd", "ner", "qfg"})
        self.assertEqual(len(set(derived.values())), len(derived))
        self.assertTrue(
            self.metadata[
                "derived_seeds_are_additional_training_seeds"
            ]
            is False
        )
        del repeated_final

    def test_export_removes_only_tss_and_strict_inference_matches(self) -> None:
        training_state = self.final.state_dict()
        inference_state = builder.export_final_inference_state(self.final)
        self.assertEqual(
            len(inference_state),
            builder.FINAL_INFERENCE_STATE_KEY_COUNT,
        )
        self.assertFalse(
            any(
                key.startswith(SURVIVAL_STATE_PREFIX)
                for key in inference_state
            )
        )
        for key in QFG_STATE_KEYS:
            self.assertTrue(
                torch.equal(inference_state[key], training_state[key]),
                msg=key,
            )
        self.assertEqual(
            set(training_state) - set(inference_state),
            set(SURVIVAL_STATE_KEYS),
        )

        inference, metadata = (
            builder.build_final_inference_model_from_training_state_dict(
                training_state,
                "SIRST3",
            )
        )
        self.assertIs(
            type(inference),
            TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
        )
        self.assertFalse(hasattr(inference, "target_survival"))
        self.assertEqual(inference.mode, "test")
        self.assertFalse(inference.training)
        self.assertTrue(metadata["strict_load"])
        self.assertEqual(
            builder.state_dict_sha256(inference_state),
            builder.state_dict_sha256(inference.state_dict()),
        )

        self.final.eval()
        self.final.mode = "test"
        images = torch.linspace(
            -1.0,
            1.0,
            steps=32 * 32,
        ).reshape(1, 1, 32, 32)
        with torch.no_grad():
            training_output = self.final(images)
            inference_output = inference(images)
        self.assertTrue(torch.equal(training_output, inference_output))
        self.final.mode = "train"
        self.final.train()

    def test_method_builder_selects_only_after_pair_construction(self) -> None:
        original = nn.Linear(1, 1)
        final = nn.Linear(1, 1)
        pair_metadata = {"paired_initialization": True}
        with mock.patch.object(
            builder,
            "build_paired_models",
            return_value=(original, final, pair_metadata),
        ) as paired:
            selected, metadata = builder.build_method_model("original")
        paired.assert_called_once_with(
            builder.TRAINING_SEED,
            dataset_name=None,
            final_with_tss=True,
        )
        self.assertIs(selected, original)
        self.assertEqual(metadata["method"], "original_scratch")

    def test_rejects_any_other_training_seed_or_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one training seed"):
            builder.build_paired_models(builder.TRAINING_SEED + 1)
        with self.assertRaises(ValueError):
            builder.build_paired_paper_models("not-a-paper-dataset")


if __name__ == "__main__":
    unittest.main()
