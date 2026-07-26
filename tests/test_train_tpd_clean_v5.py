from __future__ import annotations

import unittest

import torch

from experiments.train_tpd_clean_v5 import build_clean_v5_model
from experiments.train_tpd_pilot import build_model, model_checksum
from model.tpd_clean_v5 import (
    PRIMARY_CLEAN_V5_VARIANT,
    SUPPORTED_CLEAN_V5_VARIANTS,
    TPDCleanV5PatchEmbedding,
)


class TrainTPDCleanV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spd, cls.spd_metadata = build_model("spd", seed=42)
        cls.models = {}
        cls.metadata = {}
        for variant in SUPPORTED_CLEAN_V5_VARIANTS:
            model, metadata = build_clean_v5_model(variant, seed=42)
            cls.models[variant] = model
            cls.metadata[variant] = metadata

    def test_variants_have_paired_capacity_and_initial_state(self) -> None:
        metadata = list(self.metadata.values())
        for item in metadata:
            self.assertEqual(item["total_parameters"], 10_843_155)
            self.assertEqual(item["trainable_parameters"], 10_843_155)
            self.assertEqual(item["shallow_embedding_parameters"], 66_176)
        self.assertEqual(
            len({item["full_initialization_sha256"] for item in metadata}),
            1,
        )
        self.assertEqual(
            len({item["shared_initialization_sha256"] for item in metadata}),
            1,
        )

        full = self.models["tpd_clean_v5_full"].state_dict()
        control = self.models["tpd_clean_v5_sal_capacity"].state_dict()
        self.assertEqual(tuple(full), tuple(control))
        for key in full:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(full[key], control[key]))

    def test_only_shallow_embeddings_change_and_step_zero_is_spd(self) -> None:
        reference = self.spd.state_dict()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260726)
        inputs = torch.randn(2, 1, 64, 64, generator=generator)
        self.spd.eval()
        with torch.inference_mode():
            expected_outputs = self.spd(inputs)

        for variant, model in self.models.items():
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
                self.assertEqual(tuple(shared_candidate), tuple(shared_reference))
                for key in shared_reference:
                    self.assertTrue(
                        torch.equal(
                            shared_candidate[key],
                            shared_reference[key],
                        ),
                        msg=key,
                    )
                self.assertIsInstance(
                    model.mtc.embeddings_1,
                    TPDCleanV5PatchEmbedding,
                )
                self.assertIsInstance(
                    model.mtc.embeddings_2,
                    TPDCleanV5PatchEmbedding,
                )
                self.assertEqual(len(model.mtc.embeddings_1.blocks), 4)
                self.assertEqual(len(model.mtc.embeddings_2.blocks), 3)
                self.assertEqual(
                    self.metadata[variant]["shared_initialization_sha256"],
                    self.spd_metadata["shared_initialization_sha256"],
                )
                model.eval()
                with torch.inference_mode():
                    outputs = model(inputs)
                self.assertEqual(len(outputs), 6)
                for output, expected in zip(outputs, expected_outputs):
                    self.assertTrue(torch.equal(output, expected))

    def test_metadata_records_v5_mainline_and_formula(self) -> None:
        full = self.metadata["tpd_clean_v5_full"]
        control = self.metadata["tpd_clean_v5_sal_capacity"]
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        self.assertEqual(
            full["context_selector"],
            "positive_centered_0p5_to_1p5",
        )
        self.assertEqual(control["context_selector"], "neutral_one")
        for item in (full, control):
            self.assertEqual(
                item["candidate_family"],
                "spd_anchored_tpd_clean_v5_positive_context_selector",
            )
            self.assertEqual(item["mainline_contract"], "Keep-Context-Saliency")
            self.assertFalse(item["fourth_parallel_branch_added"])
            self.assertEqual(item["context_selector_floor"], 0.5)
            self.assertEqual(item["context_selector_ceiling"], 1.5)
            self.assertEqual(item["learned_scales_per_block"], 1)
            self.assertEqual(
                item["fusion_formula"],
                "K+S*tanh(saliency_scale*(1+0.5*context_code))",
            )
            self.assertEqual(item["zero_scale_reference"], "dense_spd_exact")

    def test_no_second_scale_relay_or_hooks(self) -> None:
        for variant, model in self.models.items():
            names = tuple(name.lower() for name, _ in model.named_modules())
            with self.subTest(variant=variant):
                self.assertFalse(any("relay" in name or "ner" in name for name in names))
                for embedding_name in ("embeddings_1", "embeddings_2"):
                    embedding = getattr(model.mtc, embedding_name)
                    for block in embedding.blocks:
                        self.assertFalse(hasattr(block, "context_scale"))
                self.assertTrue(
                    all(not module._forward_hooks for module in model.modules())
                )
                self.assertTrue(
                    all(
                        not module._forward_pre_hooks
                        for module in model.modules()
                    )
                )

    def test_full_model_strict_reload_preserves_checksum_and_outputs(self) -> None:
        source, _ = build_clean_v5_model(PRIMARY_CLEAN_V5_VARIANT, seed=101)
        with torch.no_grad():
            for embedding_name in ("embeddings_1", "embeddings_2"):
                embedding = getattr(source.mtc, embedding_name)
                for index, block in enumerate(embedding.blocks):
                    block.saliency_scale.add_(0.02 * (index + 1))
                    block.phase_compress.weight.add_(0.0001)
        state = {
            name: value.detach().cpu().clone()
            for name, value in source.state_dict().items()
        }
        checksum = model_checksum(source)
        inputs = torch.randn(2, 1, 64, 64)
        source.eval()
        with torch.inference_mode():
            expected = tuple(item.detach().clone() for item in source(inputs))

        rebuilt, _ = build_clean_v5_model(
            PRIMARY_CLEAN_V5_VARIANT,
            seed=102,
        )
        incompatible = rebuilt.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(model_checksum(rebuilt), checksum)
        rebuilt.eval()
        with torch.inference_mode():
            actual = rebuilt(inputs)
        for output, reference in zip(actual, expected):
            self.assertTrue(torch.equal(output, reference))


if __name__ == "__main__":
    unittest.main()
