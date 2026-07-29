from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.export_tpd_ner_v4_survival_to_inference import (
    EXPORT_SCHEMA,
    INFERENCE_PARAMETER_COUNT,
    INFERENCE_STATE_KEY_COUNT,
    build_inference_model_from_survival_state_dict,
    export_survival_checkpoint,
    strip_survival_state_dict,
)
from experiments.tpd_extension_warm_start import load_parent_into_extension
from model.tpd_forward_contract import evaluator_prediction
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    TPDNERV8MPRSDCHV4SCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_V4_PARENT_STATE_KEY_COUNT,
    SURVIVAL_STATE_KEYS,
    build_formal_v4_reference,
    build_formal_v4_survival_model,
)


torch.set_num_threads(1)


class V4SurvivalExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, _ = build_formal_v4_reference()
        cls.template, _ = build_formal_v4_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.reference
        del cls.template

    def _trained_head_fixture(self):
        model = copy.deepcopy(self.template)
        with torch.no_grad():
            for index, parameter in enumerate(
                model.target_survival.parameters(),
                start=1,
            ):
                parameter.fill_(0.01 * index)
        return model

    def _strict_warm_started_fixture(
        self,
        directory: Path,
    ) -> tuple[
        TPDNERV8MPRSDCHV4SCTransNet,
        torch.nn.Module,
    ]:
        reference = copy.deepcopy(self.reference)
        extension = copy.deepcopy(self.template)
        checkpoint = directory / "v4_parent.pth.tar"
        torch.save({"state_dict": reference.state_dict()}, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = load_parent_into_extension(
            parent_checkpoint=checkpoint,
            parent_model=reference,
            extension_model=extension,
            new_module_prefixes=("target_survival",),
            zero_init_prefixes=(
                "target_survival.heads.emb1.classifier",
                "target_survival.heads.emb2.classifier",
            ),
            parent_state_dict_path=("state_dict",),
            expected_parent_checkpoint_sha256=digest,
        )
        self.assertEqual(
            result.parent_state_key_count,
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertEqual(result.preserved_new_state_key_count, 4)
        return reference, extension

    def test_warm_start_tss_and_export_are_elementwise_v4_equivalent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            reference, training_model = self._strict_warm_started_fixture(
                Path(directory_text)
            )
        reference.eval()
        training_model.eval()
        inference_model, metadata = (
            build_inference_model_from_survival_state_dict(
                training_model.state_dict()
            )
        )
        self.assertIs(type(inference_model), TPDNERV8MPRSDCHV4SCTransNet)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in inference_model.parameters()
            ),
            10_854_446,
        )
        self.assertEqual(
            metadata["state_key_count"],
            544,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260729)
        images = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            reference_prediction = evaluator_prediction(reference(images))
            training_prediction = evaluator_prediction(
                training_model(images)
            )
            inference_prediction = evaluator_prediction(
                inference_model(images)
            )
        self.assertTrue(
            torch.equal(reference_prediction, training_prediction)
        )
        self.assertTrue(
            torch.equal(training_prediction, inference_prediction)
        )

    def test_strip_load_and_nonzero_heads_are_eval_invariant(self) -> None:
        training_model = self._trained_head_fixture().eval()
        training_state = training_model.state_dict()
        inference_state = strip_survival_state_dict(training_state)
        self.assertEqual(
            len(inference_state),
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertFalse(
            any(key.startswith("target_survival.") for key in inference_state)
        )

        inference_model, metadata = (
            build_inference_model_from_survival_state_dict(training_state)
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in inference_model.parameters()
            ),
            PRODUCTION_V4_RELAY_ON_PARAMETERS,
        )
        self.assertEqual(
            PRODUCTION_V4_RELAY_ON_PARAMETERS,
            INFERENCE_PARAMETER_COUNT,
        )
        self.assertEqual(
            metadata["state_key_count"],
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertEqual(
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
            INFERENCE_STATE_KEY_COUNT,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260729)
        images = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_prediction = evaluator_prediction(
                training_model(images)
            )
            inference_prediction = evaluator_prediction(
                inference_model(images)
            )
        self.assertTrue(
            torch.equal(training_prediction, inference_prediction)
        )

    def test_export_checkpoint_schema_and_strict_head_key_validation(
        self,
    ) -> None:
        training_model = self._trained_head_fixture()
        state = training_model.state_dict()
        missing = dict(state)
        missing.pop(SURVIVAL_STATE_KEYS[0])
        with self.assertRaisesRegex(ValueError, "548 state keys"):
            strip_survival_state_dict(missing)

        foreign = dict(state)
        foreign["target_survival.fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "548 state keys"):
            strip_survival_state_dict(foreign)

        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            checkpoint = directory / "survival.pth.tar"
            output = directory / "inference.pth.tar"
            torch.save({"state_dict": state}, checkpoint)
            with mock.patch(
                "experiments.export_tpd_ner_v4_survival_to_inference."
                "require_formal_survival_checkpoint_payload",
                side_effect=lambda payload: dict(payload),
            ):
                exported = export_survival_checkpoint(checkpoint, output)
            self.assertTrue(output.is_file())
            self.assertEqual(exported["schema"], EXPORT_SCHEMA)
            self.assertEqual(
                exported["inference_parameter_count"],
                10_854_446,
            )
            self.assertEqual(
                exported["inference_state_key_count"],
                544,
            )
            self.assertEqual(
                set(exported["survival_state_removed"]),
                set(SURVIVAL_STATE_KEYS),
            )
            self.assertEqual(
                len(exported["state_dict"]),
                FORMAL_V4_PARENT_STATE_KEY_COUNT,
            )
            loaded = torch.load(
                output,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(loaded["schema"], EXPORT_SCHEMA)
            self.assertFalse(
                any(
                    key.startswith("target_survival.")
                    for key in loaded["state_dict"]
                )
            )
            with self.assertRaises(FileExistsError):
                with mock.patch(
                    "experiments.export_tpd_ner_v4_survival_to_inference."
                    "require_formal_survival_checkpoint_payload",
                    side_effect=lambda payload: dict(payload),
                ):
                    export_survival_checkpoint(checkpoint, output)

    def test_export_rejects_state_only_nonformal_checkpoint(self) -> None:
        training_model = self._trained_head_fixture()
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            checkpoint = directory / "state_only.pth.tar"
            output = directory / "must_not_exist.pth.tar"
            torch.save(
                {"state_dict": training_model.state_dict()},
                checkpoint,
            )
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint lacks fields",
            ):
                export_survival_checkpoint(checkpoint, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
