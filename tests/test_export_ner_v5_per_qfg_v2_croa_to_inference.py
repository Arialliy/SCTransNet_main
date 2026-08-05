from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import three_dataset_ner_v5_per_models_seed42_v1 as registry
from experiments import train_three_dataset_ner_v5_per_tss_off_seed42 as trainer
from experiments.export_ner_v5_per_qfg_v2_croa_to_inference import (
    EXPORT_SCHEMA,
    INFERENCE_PARAMETER_COUNT,
    INFERENCE_STATE_KEY_COUNT,
    TRAINER_SCHEMA,
    TRAINING_STATE_KEY_COUNT,
    assert_training_inference_equivalent,
    build_v5_inference_model_from_training_state_dict,
    export_v5_checkpoint,
    require_v5_checkpoint_payload,
    strip_v5_tss_state_dict,
    validate_exported_v5_checkpoint,
)
from model.tpd_forward_contract import evaluator_prediction
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import V4_RELAY_VERSION
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
)
from model.tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet,
    build_formal_v5_per_qfg_v2_croa_survival_model,
)


torch.set_num_threads(1)


class NERV5PERInferenceExportTests(unittest.TestCase):
    DATASET = "NUAA-SIRST"

    @classmethod
    def setUpClass(cls) -> None:
        cls.template, _ = build_formal_v5_per_qfg_v2_croa_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.template

    def _trained_fixture(self):
        model = copy.deepcopy(self.template)
        with torch.no_grad():
            model.tpd_ner.gates["2"].weight.normal_(std=0.025)
            model.tpd_ner.dc_offsets["2"].fill_(0.05)
            for index, level in enumerate(model.tpd_qfg.levels, start=1):
                level.gate_out.weight.fill_(0.001 * index)
                level.alpha.add_(0.01 * index)
        # TSS-off requires the four registered training-only tensors to stay
        # exact zero even after the segmentation graph has trained.
        for key in SURVIVAL_STATE_KEYS:
            self.assertEqual(
                int(torch.count_nonzero(model.state_dict()[key])),
                0,
            )
        return model

    def _model_metadata(self, model: torch.nn.Module) -> dict[str, object]:
        manifest = model.architecture_manifest()
        return {
            "schema": registry.SCHEMA,
            "dataset_name": self.DATASET,
            "method": registry.METHOD,
            "recipe_id": registry.RECIPE_ID,
            "training_seed": registry.TRAINING_SEED,
            "training_graph": True,
            "state_key_count": TRAINING_STATE_KEY_COUNT,
            "target_survival_registered": True,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "tss_loss_consumes_logits": False,
            "relay_version": V5_PER_RELAY_VERSION,
            "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
            "architecture_manifest": manifest,
            "architecture_id": registry.canonical_sha256(manifest),
        }

    def _checkpoint_payload(self, model: torch.nn.Module) -> dict[str, object]:
        metadata = self._model_metadata(model)
        return {
            "schema": TRAINER_SCHEMA,
            "epoch": 200,
            "dataset": self.DATASET,
            "method": trainer.METHOD,
            "seed": trainer.TRAINING_SEED,
            "checkpoint_role": "best_miou",
            "selection_source": f"test_{self.DATASET}",
            "test_selected": True,
            "selection_is_optimistic": True,
            "test_metrics": {"miou": 0.5, "pd": 0.75, "fa": 1e-6},
            "state_dict": model.state_dict(),
            "model_metadata": metadata,
            "protocol_sha256": "a" * 64,
            "recipe": trainer.recipe_identity(
                argparse.Namespace(method=trainer.METHOD, tss_weight=0.0)
            ),
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "architecture_id": metadata["architecture_id"],
            "relay_version": V5_PER_RELAY_VERSION,
            "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
        }

    def test_registry_strip_and_exact_head_free_v5_output(self) -> None:
        training_model = self._trained_fixture().eval()
        training_state = training_model.state_dict()
        stripped = strip_v5_tss_state_dict(training_state)
        self.assertEqual(len(training_state), TRAINING_STATE_KEY_COUNT)
        self.assertEqual(len(stripped), INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(
            set(training_state) - set(stripped),
            set(SURVIVAL_STATE_KEYS),
        )
        self.assertFalse(
            any(key.startswith(SURVIVAL_STATE_PREFIX) for key in stripped)
        )

        inference_model, metadata = (
            build_v5_inference_model_from_training_state_dict(
                training_state,
                dataset_name=self.DATASET,
            )
        )
        self.assertIs(
            type(inference_model),
            TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet,
        )
        self.assertFalse(hasattr(inference_model, "target_survival"))
        self.assertEqual(len(inference_model.state_dict()), INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in inference_model.parameters()),
            INFERENCE_PARAMETER_COUNT,
        )
        self.assertFalse(
            metadata["architecture_manifest"]
            ["checkpoint_semantically_interchangeable_with_v4"]
        )
        images = torch.linspace(-0.75, 0.75, 32 * 32).reshape(1, 1, 32, 32)
        with torch.no_grad():
            training_prediction = evaluator_prediction(training_model(images))
            inference_prediction = evaluator_prediction(inference_model(images))
        self.assertTrue(torch.equal(training_prediction, inference_prediction))
        assert_training_inference_equivalent(
            training_state,
            dataset_name=self.DATASET,
            images=images,
        )

    def test_checkpoint_validator_rejects_v4_semantic_interpretation(self) -> None:
        payload = self._checkpoint_payload(self._trained_fixture())
        validated = require_v5_checkpoint_payload(payload)
        self.assertEqual(validated["schema"], TRAINER_SCHEMA)

        wrong_relay = copy.deepcopy(payload)
        manifest = wrong_relay["model_metadata"]["architecture_manifest"]
        manifest["relay_version"] = V4_RELAY_VERSION
        wrong_id = registry.canonical_sha256(manifest)
        wrong_relay["model_metadata"]["architecture_id"] = wrong_id
        wrong_relay["architecture_id"] = wrong_id
        with self.assertRaisesRegex(ValueError, "manifest.*relay_version"):
            require_v5_checkpoint_payload(wrong_relay)

        interchangeable = copy.deepcopy(payload)
        manifest = interchangeable["model_metadata"]["architecture_manifest"]
        manifest["checkpoint_semantically_interchangeable_with_v4"] = True
        wrong_id = registry.canonical_sha256(manifest)
        interchangeable["model_metadata"]["architecture_id"] = wrong_id
        interchangeable["architecture_id"] = wrong_id
        with self.assertRaisesRegex(ValueError, "semantically_interchangeable"):
            require_v5_checkpoint_payload(interchangeable)

    def test_export_is_write_once_v5_only_and_revalidates(self) -> None:
        payload = self._checkpoint_payload(self._trained_fixture())
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            checkpoint = directory / "v5_best_miou.pth.tar"
            output = directory / "v5_best_miou_inference.pth.tar"
            torch.save(payload, checkpoint)
            with mock.patch(
                "experiments.export_ner_v5_per_qfg_v2_croa_to_inference."
                "assert_training_inference_equivalent"
            ) as equivalence:
                exported = export_v5_checkpoint(checkpoint, output)
            equivalence.assert_called_once()
            self.assertTrue(output.is_file())
            self.assertEqual(exported["schema"], EXPORT_SCHEMA)
            self.assertEqual(exported["source_trainer_schema"], TRAINER_SCHEMA)
            self.assertEqual(exported["source_state_semantics"], "ner_v5_per_only")
            self.assertFalse(exported["v4_semantic_interpretation_allowed"])
            self.assertEqual(
                set(exported["tss_state_removed"]),
                set(SURVIVAL_STATE_KEYS),
            )
            self.assertEqual(len(exported["state_dict"]), INFERENCE_STATE_KEY_COUNT)
            validated = validate_exported_v5_checkpoint(
                output,
                expected_source_checkpoint=checkpoint,
            )
            self.assertTrue(validated["strict_v5_load"])
            self.assertTrue(validated["tss_state_absent"])
            self.assertFalse(validated["v4_semantic_interpretation_allowed"])
            with self.assertRaises(FileExistsError):
                export_v5_checkpoint(checkpoint, output)

    def test_state_only_or_nonzero_tss_checkpoint_is_rejected(self) -> None:
        model = self._trained_fixture()
        with self.assertRaisesRegex(ValueError, "field 'schema'"):
            require_v5_checkpoint_payload({"state_dict": model.state_dict()})

        payload = self._checkpoint_payload(model)
        state = dict(payload["state_dict"])
        state[SURVIVAL_STATE_KEYS[0]] = torch.ones_like(
            state[SURVIVAL_STATE_KEYS[0]]
        )
        payload["state_dict"] = state
        with self.assertRaisesRegex(ValueError, "exact-zero"):
            require_v5_checkpoint_payload(payload)


if __name__ == "__main__":
    unittest.main()
