from __future__ import annotations

import unittest

import torch

from experiments.train_tpd_clean_v2 import build_clean_model
from experiments.train_tpd_clean_v3 import build_clean_v3_model
from experiments.train_tpd_pilot import build_model
from model.tpd_clean_v3 import (
    PRIMARY_CLEAN_V3_VARIANT,
    SUPPORTED_CLEAN_V3_VARIANTS,
)


class TrainTPDCleanV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spd, cls.spd_metadata = build_model("spd", seed=42)
        cls.v2_full, cls.v2_metadata = build_clean_model(
            "tpd_clean_full", seed=42
        )
        cls.v3_models = {}
        cls.v3_metadata = {}
        for variant in SUPPORTED_CLEAN_V3_VARIANTS:
            model, metadata = build_clean_v3_model(variant, seed=42)
            cls.v3_models[variant] = model
            cls.v3_metadata[variant] = metadata

    def test_variants_have_identical_capacity_and_initial_state(self) -> None:
        metadata = list(self.v3_metadata.values())
        self.assertTrue(
            all(item["total_parameters"] == 10_843_475 for item in metadata)
        )
        self.assertTrue(
            all(
                item["shallow_embedding_parameters"] == 66_496
                for item in metadata
            )
        )
        self.assertEqual(
            len({item["full_initialization_sha256"] for item in metadata}), 1
        )
        self.assertEqual(
            len({item["shared_initialization_sha256"] for item in metadata}),
            1,
        )

    def test_v3_initial_state_is_strictly_compatible_with_v2_full(self) -> None:
        v2_state = self.v2_full.state_dict()
        v3_state = self.v3_models[PRIMARY_CLEAN_V3_VARIANT].state_dict()
        self.assertEqual(tuple(v3_state), tuple(v2_state))
        for key in v2_state:
            with self.subTest(key=key):
                self.assertEqual(v3_state[key].shape, v2_state[key].shape)
                torch.testing.assert_close(
                    v3_state[key], v2_state[key], rtol=0.0, atol=0.0
                )
        incompatible = self.v3_models[
            PRIMARY_CLEAN_V3_VARIANT
        ].load_state_dict(v2_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_step_zero_full_network_is_exactly_spd(self) -> None:
        inputs = torch.randn(2, 1, 64, 64)
        self.spd.eval()
        with torch.no_grad():
            spd_outputs = self.spd(inputs)
            self.assertEqual(len(spd_outputs), 6)
            for variant, model in self.v3_models.items():
                model.eval()
                outputs = model(inputs)
                with self.subTest(variant=variant):
                    self.assertEqual(len(outputs), 6)
                    for output, reference in zip(outputs, spd_outputs):
                        torch.testing.assert_close(
                            output, reference, rtol=0.0, atol=0.0
                        )

    def test_only_shallow_embeddings_change_from_spd_builder(self) -> None:
        reference = self.spd.state_dict()
        candidate = self.v3_models[PRIMARY_CLEAN_V3_VARIANT].state_dict()
        shared_reference = {
            key: value
            for key, value in reference.items()
            if not key.startswith(("mtc.embeddings_1.", "mtc.embeddings_2."))
        }
        shared_candidate = {
            key: value
            for key, value in candidate.items()
            if not key.startswith(("mtc.embeddings_1.", "mtc.embeddings_2."))
        }
        self.assertEqual(tuple(shared_candidate), tuple(shared_reference))
        for key in shared_reference:
            torch.testing.assert_close(
                shared_candidate[key],
                shared_reference[key],
                rtol=0.0,
                atol=0.0,
            )

    def test_metadata_records_mainline_and_control_semantics(self) -> None:
        full = self.v3_metadata["tpd_clean_v3_full"]
        control = self.v3_metadata["tpd_clean_v3_sal_capacity"]
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        self.assertEqual(full["mainline_contract"], "Keep-Context-Saliency")
        self.assertEqual(full["context_code"], "centered_spatial_rms_tanh")
        self.assertEqual(control["context_code"], "constant_one")
        for metadata in (full, control):
            self.assertFalse(metadata["fourth_parallel_branch_added"])
            self.assertEqual(metadata["zero_scale_reference"], "dense_spd_exact")
            self.assertEqual(
                metadata["shared_initialization_sha256"],
                self.spd_metadata["shared_initialization_sha256"],
            )

    def test_no_runtime_relay_or_forward_hooks_are_present(self) -> None:
        for variant, model in self.v3_models.items():
            names = tuple(name.lower() for name, _ in model.named_modules())
            with self.subTest(variant=variant):
                self.assertFalse(any("relay" in name for name in names))
                self.assertFalse(any("ner" in name for name in names))
                self.assertTrue(
                    all(not module._forward_hooks for module in model.modules())
                )
                self.assertTrue(
                    all(
                        not module._forward_pre_hooks
                        for module in model.modules()
                    )
                )


if __name__ == "__main__":
    unittest.main()
