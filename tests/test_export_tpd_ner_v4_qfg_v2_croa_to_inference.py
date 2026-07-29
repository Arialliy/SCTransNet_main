from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.export_tpd_ner_v4_qfg_v2_croa_to_inference import (
    EXPORT_SCHEMA,
    INFERENCE_PARAMETER_COUNT,
    INFERENCE_STATE_KEY_COUNT,
    TRAINING_STATE_KEY_COUNT,
    build_inference_model_from_training_state_dict,
    export_qfg_checkpoint,
    strip_survival_state_dict,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


torch.set_num_threads(1)


class QFGV2CROAExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template, _ = build_formal_v4_qfg_v2_croa_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.template

    def _trained_fixture(self):
        model = copy.deepcopy(self.template)
        with torch.no_grad():
            for index, parameter in enumerate(
                model.target_survival.parameters(),
                start=1,
            ):
                parameter.fill_(0.01 * index)
            for level_index, level in enumerate(model.tpd_qfg.levels):
                level.gate_out.weight.fill_(0.0025 * (level_index + 1))
                level.alpha.add_(0.05 * (level_index + 1))
        return model

    def test_strip_preserves_all_qfg_and_removes_only_survival(self) -> None:
        training_state = self._trained_fixture().state_dict()
        self.assertEqual(len(training_state), TRAINING_STATE_KEY_COUNT)
        inference_state = strip_survival_state_dict(training_state)
        self.assertEqual(len(inference_state), INFERENCE_STATE_KEY_COUNT)
        self.assertFalse(
            any(
                key.startswith(SURVIVAL_STATE_PREFIX)
                for key in inference_state
            )
        )
        self.assertEqual(
            {
                key
                for key in inference_state
                if key.startswith(QFG_STATE_PREFIX)
            },
            set(QFG_STATE_KEYS),
        )
        for key in QFG_STATE_KEYS:
            self.assertTrue(
                torch.equal(inference_state[key], training_state[key]),
                msg=key,
            )

    def test_strict_head_free_load_has_identical_legacy_eval_output(self) -> None:
        training_model = self._trained_fixture().eval()
        inference_model, metadata = (
            build_inference_model_from_training_state_dict(
                training_model.state_dict()
            )
        )
        self.assertIs(
            type(inference_model),
            TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
        )
        self.assertFalse(hasattr(inference_model, "target_survival"))
        self.assertEqual(
            sum(parameter.numel() for parameter in inference_model.parameters()),
            INFERENCE_PARAMETER_COUNT,
        )
        self.assertEqual(len(inference_model.state_dict()), INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(metadata["state_key_count"], INFERENCE_STATE_KEY_COUNT)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260729)
        images = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_output = training_model(images)
            inference_output = inference_model(images)
        self.assertIsInstance(training_output, tuple)
        self.assertIsInstance(inference_output, tuple)
        self.assertEqual(len(training_output), 6)
        self.assertEqual(len(inference_output), 6)
        for index, (actual, expected) in enumerate(
            zip(inference_output, training_output)
        ):
            self.assertTrue(torch.equal(actual, expected), msg=index)

    def test_strip_rejects_missing_foreign_or_incomplete_state(self) -> None:
        state = dict(self.template.state_dict())
        missing_survival = dict(state)
        missing_survival.pop(SURVIVAL_STATE_KEYS[0])
        with self.assertRaisesRegex(ValueError, "568 training state keys"):
            strip_survival_state_dict(missing_survival)

        missing_qfg = dict(state)
        missing_qfg.pop(QFG_STATE_KEYS[0])
        missing_qfg["fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "twenty QFG keys"):
            strip_survival_state_dict(missing_qfg)

        foreign_survival = dict(state)
        foreign_survival["target_survival.fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "568 training state keys"):
            strip_survival_state_dict(foreign_survival)

    def test_export_is_write_once_and_contains_strict_head_free_state(
        self,
    ) -> None:
        training_model = self._trained_fixture()
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            checkpoint = directory / "qfg.pth.tar"
            output = directory / "qfg_inference.pth.tar"
            torch.save(
                {
                    "state_dict": training_model.state_dict(),
                    "checkpoint_role": "best_validation_pd_primary",
                    "checkpoint_identity": {"schema": "checkpoint"},
                    "run_identity": {"schema": "run"},
                },
                checkpoint,
            )
            with mock.patch(
                "experiments.export_tpd_ner_v4_qfg_v2_croa_to_inference."
                "require_formal_qfg_checkpoint_payload",
                side_effect=lambda payload: dict(payload),
            ):
                exported = export_qfg_checkpoint(checkpoint, output)
            self.assertTrue(output.is_file())
            self.assertEqual(exported["schema"], EXPORT_SCHEMA)
            self.assertEqual(
                exported["inference_parameter_count"],
                INFERENCE_PARAMETER_COUNT,
            )
            self.assertEqual(
                exported["inference_state_key_count"],
                INFERENCE_STATE_KEY_COUNT,
            )
            self.assertEqual(
                set(exported["survival_state_removed"]),
                set(SURVIVAL_STATE_KEYS),
            )
            self.assertEqual(
                set(exported["qfg_state_preserved"]),
                set(QFG_STATE_KEYS),
            )
            self.assertEqual(
                len(exported["state_dict"]),
                INFERENCE_STATE_KEY_COUNT,
            )
            self.assertFalse(
                any(
                    key.startswith(SURVIVAL_STATE_PREFIX)
                    for key in exported["state_dict"]
                )
            )
            self.assertEqual(
                {
                    key
                    for key in exported["state_dict"]
                    if key.startswith(QFG_STATE_PREFIX)
                },
                set(QFG_STATE_KEYS),
            )
            loaded = torch.load(
                output,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(loaded["schema"], EXPORT_SCHEMA)
            with self.assertRaises(FileExistsError):
                with mock.patch(
                    "experiments.export_tpd_ner_v4_qfg_v2_croa_to_inference."
                    "require_formal_qfg_checkpoint_payload",
                    side_effect=lambda payload: dict(payload),
                ):
                    export_qfg_checkpoint(checkpoint, output)

    def test_export_rejects_unvalidated_state_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            checkpoint = directory / "state_only.pth.tar"
            output = directory / "must_not_exist.pth.tar"
            torch.save(
                {"state_dict": self.template.state_dict()},
                checkpoint,
            )
            with self.assertRaises((ValueError, RuntimeError, TypeError)):
                export_qfg_checkpoint(checkpoint, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
