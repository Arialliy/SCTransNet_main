from __future__ import annotations

import unittest

import torch

from experiments.train_tpd_clean_v6 import (
    CAPACITY_HEADROOM_FORMULA,
    CONTEXT_CODE_FORMULA,
    FULL_HEADROOM_FORMULA,
    FUSION_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
    SHALLOW_EMBEDDING_PARAMETERS,
    TOTAL_PARAMETERS,
    build_clean_v6_model,
)
from experiments.train_tpd_pilot import build_model, model_checksum
from model.tpd_clean_v6 import (
    PRIMARY_CLEAN_V6_VARIANT,
    SUPPORTED_CLEAN_V6_VARIANTS,
    TPDCleanV6PatchEmbedding,
)


class TrainTPDCleanV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spd, cls.spd_metadata = build_model("spd", seed=42)
        cls.models = {}
        cls.metadata = {}
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            model, metadata = build_clean_v6_model(variant, seed=42)
            cls.models[variant] = model
            cls.metadata[variant] = metadata

    def test_builder_registers_only_the_fixed_v6_candidate_pair(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V6_VARIANTS,
            (
                "tpd_clean_v6_full",
                "tpd_clean_v6_phase_capacity",
            ),
        )
        self.assertEqual(
            PRIMARY_CLEAN_V6_VARIANT,
            "tpd_clean_v6_full",
        )
        self.assertEqual(set(self.models), set(SUPPORTED_CLEAN_V6_VARIANTS))
        for model in self.models.values():
            self.assertIsInstance(
                model.mtc.embeddings_1,
                TPDCleanV6PatchEmbedding,
            )
            self.assertIsInstance(
                model.mtc.embeddings_2,
                TPDCleanV6PatchEmbedding,
            )
            self.assertEqual(len(model.mtc.embeddings_1.blocks), 4)
            self.assertEqual(len(model.mtc.embeddings_2.blocks), 3)

    def test_variants_have_paired_capacity_keys_and_initial_state(self) -> None:
        metadata = list(self.metadata.values())
        for item in metadata:
            self.assertEqual(item["total_parameters"], TOTAL_PARAMETERS)
            self.assertEqual(item["trainable_parameters"], TOTAL_PARAMETERS)
            self.assertEqual(
                item["shallow_embedding_parameters"],
                SHALLOW_EMBEDDING_PARAMETERS,
            )
        self.assertEqual(
            len({item["full_initialization_sha256"] for item in metadata}),
            1,
        )
        self.assertEqual(
            len({item["shared_initialization_sha256"] for item in metadata}),
            1,
        )

        full = self.models["tpd_clean_v6_full"].state_dict()
        control = self.models[
            "tpd_clean_v6_phase_capacity"
        ].state_dict()
        self.assertEqual(tuple(full), tuple(control))
        for key in full:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(full[key], control[key]))

    def test_only_shallow_embeddings_change_and_step_zero_is_exact_spd(
        self,
    ) -> None:
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

    def test_metadata_records_phase_tied_zero_mean_gain_kcs_formula(
        self,
    ) -> None:
        full = self.metadata["tpd_clean_v6_full"]
        control = self.metadata["tpd_clean_v6_phase_capacity"]
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        self.assertEqual(
            full["context_headroom_formula"],
            FULL_HEADROOM_FORMULA,
        )
        self.assertEqual(
            control["context_headroom_formula"],
            CAPACITY_HEADROOM_FORMULA,
        )
        self.assertEqual(
            full["context_modulation"],
            "half_centered_context_code",
        )
        self.assertEqual(control["context_modulation"], "zero")
        self.assertEqual(
            full["context_headroom"],
            "one_plus_half_one_minus_abs_scale_times_modulation",
        )
        self.assertEqual(control["context_headroom"], "neutral_one")
        self.assertEqual(
            full["context_code"],
            (
                "phase_aligned_centered_spatial_rms_eps_tanh_"
                "formal_amp_off_fp32"
            ),
        )
        self.assertEqual(
            control["context_code"],
            "phase_aligned_context_computed_but_modulation_zero",
        )
        self.assertEqual(
            full["fusion_formula"],
            (
                "K+Sa*(a*(1+0.5*(1-abs(a))*V));"
                "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
            ),
        )
        self.assertEqual(
            control["fusion_formula"],
            "K+Sa*tanh(saliency_scale)",
        )
        for item in (full, control):
            self.assertEqual(
                item["candidate_family"],
                "spd_anchored_tpd_clean_v6_phase_tied_kcs_zero_mean_gain",
            )
            self.assertEqual(item["mainline_contract"], "Keep-Context-Saliency")
            self.assertEqual(
                item["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertTrue(item["kcs_only"])
            self.assertFalse(item["fourth_parallel_branch_added"])
            self.assertEqual(
                item["replaced_embeddings"],
                ("mtc.embeddings_1", "mtc.embeddings_2"),
            )
            self.assertTrue(item["context_modulates_saliency_only"])
            self.assertEqual(
                item["phase_tied_projection"],
                "sum_keep_weights_over_four_contiguous_phases",
            )
            self.assertEqual(
                item["phase_tied_projection_formula"],
                PHASE_TIED_PROJECTION_FORMULA,
            )
            self.assertEqual(
                item["pixel_unshuffle_channel_order"],
                "input_channel_major_four_phases_contiguous",
            )
            self.assertEqual(item["derived_projection_parameters"], 0)
            self.assertEqual(item["derived_projection_buffers"], 0)
            self.assertEqual(
                item["context_code_formula"],
                CONTEXT_CODE_FORMULA,
            )
            self.assertEqual(
                item["context_modulation_spatial_mean"],
                "zero_up_to_fp32_roundoff",
            )
            self.assertEqual(
                item["context_headroom_spatial_mean"],
                "one_up_to_fp32_roundoff",
            )
            self.assertFalse(item["residual_mean_preserving"])
            self.assertEqual(item["fusion_equation"], FUSION_FORMULA)
            self.assertEqual(item["learned_scales_per_block"], 1)
            self.assertEqual(
                item["scale_parameter"],
                "per_channel_saliency_scale",
            )
            self.assertEqual(item["headroom_bound"], "0.5<=H<=1.5")
            self.assertEqual(item["coefficient_bound"], "abs(a*H)<=1")
            self.assertEqual(item["residual_bound"], "abs(R)<=abs(Sa)")
            self.assertEqual(item["zero_saliency_reference"], "R=0")
            self.assertEqual(item["zero_scale_reference"], "dense_spd_exact")
            self.assertEqual(
                item["projection_precision"],
                "float32_in_formal_amp_off_path",
            )
            self.assertEqual(
                item["context_precision"],
                "float32_in_formal_amp_off_path",
            )
            self.assertEqual(
                item["coefficient_precision"],
                "float32_in_formal_amp_off_path",
            )
            self.assertEqual(item["residual_output_dtype"], "feature_dtype")

    def test_no_projection_parameters_fourth_branch_relay_or_hooks(self) -> None:
        for variant, model in self.models.items():
            names = tuple(name.lower() for name, _ in model.named_modules())
            with self.subTest(variant=variant):
                self.assertFalse(
                    any("relay" in name or "ner" in name for name in names)
                )
                for embedding_name in ("embeddings_1", "embeddings_2"):
                    embedding = getattr(model.mtc, embedding_name)
                    self.assertEqual(tuple(embedding.named_buffers()), ())
                    for block in embedding.blocks:
                        self.assertEqual(
                            set(dict(block.named_parameters())),
                            {
                                "saliency_scale",
                                "phase_compress.weight",
                                "phase_compress.bias",
                            },
                        )
                self.assertTrue(
                    all(not module._forward_hooks for module in model.modules())
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
        source, _ = build_clean_v6_model(PRIMARY_CLEAN_V6_VARIANT, seed=101)
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

        rebuilt, _ = build_clean_v6_model(
            PRIMARY_CLEAN_V6_VARIANT,
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
