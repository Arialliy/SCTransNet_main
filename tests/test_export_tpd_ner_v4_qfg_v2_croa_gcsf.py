from __future__ import annotations

import copy
import unittest

import torch

from experiments.export_tpd_ner_v4_qfg_v2_croa_gcsf_to_inference import (
    EXPORT_SCHEMA,
    INFERENCE_PARAMETER_COUNT,
    INFERENCE_STATE_KEY_COUNT,
    TRAINING_STATE_KEY_COUNT,
    build_gcsf_inference_model_from_training_state_dict,
    export_gcsf_training_payload_to_inference,
    strip_gcsf_survival_state_dict,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf import (
    GCSF_STATE_KEYS,
    GCSF_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_gcsf_survival_model,
    load_formal_qfg_v2_croa_state_as_zero_gcsf_extension,
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


class FormalGCSFExportAndExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.current, _ = build_formal_v4_qfg_v2_croa_survival_model()
        cls.gcsf, _ = build_formal_v4_qfg_v2_croa_gcsf_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.current
        del cls.gcsf

    def test_strict_model_only_extension_allows_exactly_four_zero_keys(self) -> None:
        source = {
            name: value.detach().clone()
            for name, value in self.current.state_dict().items()
        }
        first_float = next(
            name for name, value in source.items() if value.is_floating_point()
        )
        source[first_float].add_(0.125)
        target = copy.deepcopy(self.gcsf)
        report = load_formal_qfg_v2_croa_state_as_zero_gcsf_extension(
            target,
            source,
        )
        self.assertEqual(
            report["load_mode"],
            "strict_model_only_four_key_zero_extension",
        )
        self.assertEqual(report["new_state_key_count"], 4)
        self.assertEqual(set(report["new_state_keys"]), set(GCSF_STATE_KEYS))
        self.assertFalse(report["formal_training_warm_start_authorized"])
        target_state = target.state_dict()
        for name, expected in source.items():
            with self.subTest(shared_state=name):
                self.assertTrue(torch.equal(target_state[name], expected))
        for name in GCSF_STATE_KEYS:
            self.assertEqual(int(torch.count_nonzero(target_state[name])), 0)

        missing = dict(source)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "exact current-model state"):
            load_formal_qfg_v2_croa_state_as_zero_gcsf_extension(
                copy.deepcopy(self.gcsf),
                missing,
            )
        foreign = dict(source)
        foreign["global_skip_fusion.fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "exact current-model state"):
            load_formal_qfg_v2_croa_state_as_zero_gcsf_extension(
                copy.deepcopy(self.gcsf),
                foreign,
            )

    def test_export_removes_only_tss_and_preserves_qfg_and_gcsf(self) -> None:
        training = copy.deepcopy(self.gcsf)
        with torch.no_grad():
            for level, parameter in enumerate(
                training.global_skip_fusion.reallocation_logits,
                start=1,
            ):
                parameter.fill_(0.03 * level)
            for index, parameter in enumerate(
                training.target_survival.parameters(),
                start=1,
            ):
                parameter.fill_(0.01 * index)
        training_state = training.state_dict()
        self.assertEqual(len(training_state), TRAINING_STATE_KEY_COUNT)
        inference_state = strip_gcsf_survival_state_dict(training_state)
        self.assertEqual(len(inference_state), INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(
            {
                key
                for key in inference_state
                if key.startswith(GCSF_STATE_PREFIX)
            },
            set(GCSF_STATE_KEYS),
        )
        self.assertEqual(
            {
                key
                for key in inference_state
                if key.startswith(QFG_STATE_PREFIX)
            },
            set(QFG_STATE_KEYS),
        )
        self.assertFalse(
            any(
                key.startswith(SURVIVAL_STATE_PREFIX)
                for key in inference_state
            )
        )
        for name in (*GCSF_STATE_KEYS, *QFG_STATE_KEYS):
            self.assertTrue(torch.equal(inference_state[name], training_state[name]))

        inference, metadata = (
            build_gcsf_inference_model_from_training_state_dict(training_state)
        )
        self.assertIs(
            type(inference),
            TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
        )
        self.assertFalse(hasattr(inference, "target_survival"))
        self.assertEqual(metadata["state_key_count"], INFERENCE_STATE_KEY_COUNT)
        self.assertEqual(
            sum(parameter.numel() for parameter in inference.parameters()),
            INFERENCE_PARAMETER_COUNT,
        )

        training.eval()
        inference.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026080502)
        images = torch.randn(1, 1, 32, 32, generator=generator)
        with torch.no_grad():
            training_output = training(images)
            inference_output = inference(images)
        self.assertEqual(len(training_output), 6)
        self.assertEqual(len(inference_output), 6)
        for index, (actual, expected) in enumerate(
            zip(inference_output, training_output)
        ):
            with self.subTest(export_output=index):
                self.assertTrue(torch.equal(actual, expected))

        payload = export_gcsf_training_payload_to_inference(
            {
                "state_dict": training_state,
                "checkpoint_role": "best_validation_miou",
                "checkpoint_identity": {"schema": "checkpoint"},
                "run_identity": {"schema": "run"},
            }
        )
        self.assertEqual(payload["schema"], EXPORT_SCHEMA)
        self.assertEqual(
            set(payload["survival_state_removed"]),
            set(SURVIVAL_STATE_KEYS),
        )
        self.assertEqual(
            set(payload["gcsf_state_preserved"]),
            set(GCSF_STATE_KEYS),
        )
        self.assertEqual(
            set(payload["qfg_state_preserved"]),
            set(QFG_STATE_KEYS),
        )

    def test_export_rejects_missing_or_foreign_training_state(self) -> None:
        state = dict(self.gcsf.state_dict())
        missing_tss = dict(state)
        missing_tss.pop(SURVIVAL_STATE_KEYS[0])
        with self.assertRaisesRegex(ValueError, "572 training state keys"):
            strip_gcsf_survival_state_dict(missing_tss)

        missing_gcsf = dict(state)
        missing_gcsf.pop(GCSF_STATE_KEYS[0])
        missing_gcsf["fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "four GCSF keys"):
            strip_gcsf_survival_state_dict(missing_gcsf)

        foreign_tss = dict(state)
        foreign_tss["target_survival.fabricated"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "572 training state keys"):
            strip_gcsf_survival_state_dict(foreign_tss)


if __name__ == "__main__":
    unittest.main()
