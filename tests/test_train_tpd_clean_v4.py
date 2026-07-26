from __future__ import annotations

import unittest

import torch

from experiments.train_tpd_clean_v3 import build_clean_v3_model
from experiments.train_tpd_clean_v4 import build_clean_v4_model
from experiments.train_tpd_pilot import build_model, model_checksum
from model.tpd_clean_v3 import PRIMARY_CLEAN_V3_VARIANT
from model.tpd_clean_v4 import (
    PRIMARY_CLEAN_V4_VARIANT,
    SUPPORTED_CLEAN_V4_VARIANTS,
    TPDCleanV4PatchEmbedding,
)


class TrainTPDCleanV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spd, cls.spd_metadata = build_model("spd", seed=42)
        cls.v3_full, cls.v3_metadata = build_clean_v3_model(
            PRIMARY_CLEAN_V3_VARIANT,
            seed=42,
        )
        cls.v4_models = {}
        cls.v4_metadata = {}
        for variant in SUPPORTED_CLEAN_V4_VARIANTS:
            model, metadata = build_clean_v4_model(variant, seed=42)
            cls.v4_models[variant] = model
            cls.v4_metadata[variant] = metadata

    def test_variants_have_exact_paired_capacity_and_initial_state(self) -> None:
        metadata = list(self.v4_metadata.values())
        self.assertTrue(
            all(item["total_parameters"] == 10_843_475 for item in metadata)
        )
        self.assertTrue(
            all(
                item["trainable_parameters"] == 10_843_475
                for item in metadata
            )
        )
        self.assertTrue(
            all(
                item["shallow_embedding_parameters"] == 66_496
                for item in metadata
            )
        )
        self.assertEqual(
            len(
                {
                    item["full_initialization_sha256"]
                    for item in metadata
                }
            ),
            1,
        )
        self.assertEqual(
            len(
                {
                    item["shared_initialization_sha256"]
                    for item in metadata
                }
            ),
            1,
        )

        full_state = self.v4_models[
            "tpd_clean_v4_full"
        ].state_dict()
        control_state = self.v4_models[
            "tpd_clean_v4_sal_capacity"
        ].state_dict()
        self.assertEqual(tuple(full_state), tuple(control_state))
        for key in full_state:
            with self.subTest(key=key):
                self.assertEqual(
                    full_state[key].shape,
                    control_state[key].shape,
                )
                self.assertEqual(
                    full_state[key].dtype,
                    control_state[key].dtype,
                )
                self.assertTrue(
                    torch.equal(full_state[key], control_state[key])
                )

    def test_v4_initial_state_is_strictly_anchored_to_v3(self) -> None:
        v3_state = self.v3_full.state_dict()
        v4_model = self.v4_models[PRIMARY_CLEAN_V4_VARIANT]
        v4_state = v4_model.state_dict()
        self.assertEqual(tuple(v4_state), tuple(v3_state))
        for key in v3_state:
            with self.subTest(key=key):
                self.assertEqual(v4_state[key].shape, v3_state[key].shape)
                self.assertEqual(v4_state[key].dtype, v3_state[key].dtype)
                self.assertTrue(torch.equal(v4_state[key], v3_state[key]))
        incompatible = v4_model.load_state_dict(v3_state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_only_shallow_embeddings_change_from_spd_builder(self) -> None:
        reference = self.spd.state_dict()
        for variant, model in self.v4_models.items():
            candidate = model.state_dict()
            shared_reference = {
                key: value
                for key, value in reference.items()
                if not key.startswith(
                    ("mtc.embeddings_1.", "mtc.embeddings_2.")
                )
            }
            shared_candidate = {
                key: value
                for key, value in candidate.items()
                if not key.startswith(
                    ("mtc.embeddings_1.", "mtc.embeddings_2.")
                )
            }
            with self.subTest(variant=variant):
                self.assertEqual(
                    tuple(shared_candidate),
                    tuple(shared_reference),
                )
                self.assertIsInstance(
                    model.mtc.embeddings_1,
                    TPDCleanV4PatchEmbedding,
                )
                self.assertIsInstance(
                    model.mtc.embeddings_2,
                    TPDCleanV4PatchEmbedding,
                )
                self.assertEqual(len(model.mtc.embeddings_1.blocks), 4)
                self.assertEqual(len(model.mtc.embeddings_2.blocks), 3)
                self.assertEqual(
                    self.v4_metadata[variant][
                        "shared_initialization_sha256"
                    ],
                    self.spd_metadata["shared_initialization_sha256"],
                )
            for key in shared_reference:
                torch.testing.assert_close(
                    shared_candidate[key],
                    shared_reference[key],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_step_zero_full_network_is_exactly_spd(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026)
        inputs = torch.randn(
            2,
            1,
            64,
            64,
            generator=generator,
        )
        self.spd.eval()
        with torch.inference_mode():
            spd_outputs = self.spd(inputs)
            self.assertEqual(len(spd_outputs), 6)
            for variant, model in self.v4_models.items():
                model.eval()
                outputs = model(inputs)
                with self.subTest(variant=variant):
                    self.assertEqual(len(outputs), 6)
                    for output, reference in zip(outputs, spd_outputs):
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=0.0,
                            atol=0.0,
                        )

    def test_metadata_records_frozen_v4_semantics(self) -> None:
        full = self.v4_metadata["tpd_clean_v4_full"]
        control = self.v4_metadata["tpd_clean_v4_sal_capacity"]
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        self.assertEqual(
            full["context_code"],
            "centered_spatial_rms_tanh_fp32",
        )
        self.assertEqual(control["context_code"], "constant_one")
        for metadata in (full, control):
            self.assertEqual(
                metadata["candidate_family"],
                "spd_anchored_tpd_clean_v4_single_logit_kcs",
            )
            self.assertEqual(
                metadata["mainline_contract"],
                "Keep-Context-Saliency",
            )
            self.assertFalse(metadata["fourth_parallel_branch_added"])
            self.assertEqual(
                metadata["fusion_support"],
                "single_bounded_saliency_logit",
            )
            self.assertEqual(
                metadata["fusion_formula"],
                (
                    "K+S*tanh(saliency_scale"
                    "+0.5*tanh(context_scale)*context_code)"
                ),
            )
            self.assertEqual(metadata["context_logit_limit"], 0.5)
            self.assertEqual(
                metadata["residual_bound"],
                "absolute_residual_at_most_absolute_saliency",
            )
            self.assertEqual(
                metadata["zero_scale_reference"],
                "dense_spd_exact",
            )

    def test_no_runtime_relay_or_forward_hooks_are_present(self) -> None:
        for variant, model in self.v4_models.items():
            names = tuple(name.lower() for name, _ in model.named_modules())
            with self.subTest(variant=variant):
                self.assertFalse(any("relay" in name for name in names))
                self.assertFalse(any("ner" in name for name in names))
                self.assertTrue(
                    all(
                        not module._forward_hooks
                        for module in model.modules()
                    )
                )
                self.assertTrue(
                    all(
                        not module._forward_pre_hooks
                        for module in model.modules()
                    )
                )

    def test_full_model_strict_reload_preserves_checksum_and_outputs(
        self,
    ) -> None:
        source, _ = build_clean_v4_model(
            PRIMARY_CLEAN_V4_VARIANT,
            seed=101,
        )
        with torch.no_grad():
            for embedding_name in ("embeddings_1", "embeddings_2"):
                embedding = getattr(source.mtc, embedding_name)
                for block_index, block in enumerate(embedding.blocks):
                    block.saliency_scale.add_(0.01 * (block_index + 1))
                    block.context_scale.sub_(0.015 * (block_index + 1))
                    block.phase_compress.weight.add_(0.0001)
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in source.state_dict().items()
        }
        source_checksum = model_checksum(source)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(303)
        inputs = torch.randn(
            2,
            1,
            64,
            64,
            generator=generator,
        )
        source.eval()
        with torch.inference_mode():
            source_outputs = tuple(
                output.detach().clone() for output in source(inputs)
            )

        rebuilt, _ = build_clean_v4_model(
            PRIMARY_CLEAN_V4_VARIANT,
            seed=102,
        )
        incompatible = rebuilt.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(model_checksum(rebuilt), source_checksum)
        rebuilt.eval()
        with torch.inference_mode():
            rebuilt_outputs = rebuilt(inputs)
        self.assertEqual(len(rebuilt_outputs), 6)
        for output, reference in zip(rebuilt_outputs, source_outputs):
            torch.testing.assert_close(
                output,
                reference,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
