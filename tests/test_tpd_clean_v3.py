from __future__ import annotations

import io
import unittest

import torch

from experiments.train_tpd_pilot import weights_init_kaiming
from model.tpd import SPDPatchEmbedding
from model.tpd_clean_v3 import (
    PRIMARY_CLEAN_V3_VARIANT,
    SUPPORTED_CLEAN_V3_VARIANTS,
    TPDCleanV3Block,
    TPDCleanV3PatchEmbedding,
    build_clean_v3_patch_embedding,
    clean_v3_variant_spec,
    parameter_count,
)


class TPDCleanV3Tests(unittest.TestCase):
    def test_frozen_candidate_matrix_and_primary(self) -> None:
        self.assertEqual(len(SUPPORTED_CLEAN_V3_VARIANTS), 2)
        self.assertEqual(PRIMARY_CLEAN_V3_VARIANT, "tpd_clean_v3_full")
        self.assertEqual(
            {
                clean_v3_variant_spec(variant)["context_code"]
                for variant in SUPPORTED_CLEAN_V3_VARIANTS
            },
            {
                "centered_spatial_rms_tanh",
                "constant_one",
            },
        )
        self.assertTrue(
            clean_v3_variant_spec(PRIMARY_CLEAN_V3_VARIANT)[
                "primary_candidate"
            ]
        )

    def test_all_variants_match_required_shapes(self) -> None:
        cases = ((32, 16, 192, 288), (64, 8, 96, 144))
        for channels, stride, height, width in cases:
            inputs = torch.randn(2, channels, height, width)
            expected = (2, channels, height // stride, width // stride)
            for variant in SUPPORTED_CLEAN_V3_VARIANTS:
                with self.subTest(variant=variant, channels=channels):
                    output = build_clean_v3_patch_embedding(
                        variant, channels, stride
                    )(inputs)
                    self.assertEqual(tuple(output.shape), expected)

    def test_zero_scales_are_exactly_spd_equivalent(self) -> None:
        channels = 4
        stride = 8
        spd = SPDPatchEmbedding(channels, stride)
        clean = TPDCleanV3PatchEmbedding(
            channels,
            stride,
            use_context_code=True,
        )
        with torch.no_grad():
            for spd_block, clean_block in zip(spd.blocks, clean.blocks):
                clean_block.phase_compress.weight.copy_(
                    spd_block.phase_compress.weight
                )
                clean_block.phase_compress.bias.copy_(
                    spd_block.phase_compress.bias
                )
        inputs = torch.randn(2, channels, 32, 40)
        torch.testing.assert_close(clean(inputs), spd(inputs), rtol=0.0, atol=0.0)

    def test_formal_initializer_preserves_spd_starting_point(self) -> None:
        channels = 4
        stride = 8
        torch.manual_seed(17)
        spd = SPDPatchEmbedding(channels, stride)
        torch.manual_seed(17)
        clean = TPDCleanV3PatchEmbedding(
            channels,
            stride,
            use_context_code=True,
        )
        torch.manual_seed(29)
        spd.apply(weights_init_kaiming)
        torch.manual_seed(29)
        clean.apply(weights_init_kaiming)

        spd_x = torch.randn(2, channels, 32, 40)
        clean_x = spd_x.clone()
        for spd_block, clean_block in zip(spd.blocks, clean.blocks):
            self.assertEqual(
                int(torch.count_nonzero(clean_block.context_scale)), 0
            )
            self.assertEqual(
                int(torch.count_nonzero(clean_block.saliency_scale)), 0
            )
            torch.testing.assert_close(
                clean_block.phase_compress.weight,
                spd_block.phase_compress.weight,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                clean_block.phase_compress.bias,
                spd_block.phase_compress.bias,
                rtol=0.0,
                atol=0.0,
            )
            spd_x = spd_block(spd_x)
            clean_x = clean_block(clean_x)
            torch.testing.assert_close(clean_x, spd_x, rtol=0.0, atol=0.0)

    def test_every_candidate_receives_finite_zero_scale_gradients(self) -> None:
        spatial_weight = torch.tensor(
            [[[[1.0, -0.5], [0.25, 1.5]]]], dtype=torch.float32
        )
        for variant in SUPPORTED_CLEAN_V3_VARIANTS:
            spec = clean_v3_variant_spec(variant)
            block = TPDCleanV3Block(
                channels=2,
                activate=False,
                use_context_code=spec["context_code"]
                == "centered_spatial_rms_tanh",
            )
            with torch.no_grad():
                block.phase_compress.weight.fill_(0.125)
                block.phase_compress.bias.zero_()
            inputs = torch.tensor(
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 8.0],
                            [4.0, 9.0, 2.0, 1.0],
                            [7.0, 1.0, 5.0, 2.0],
                            [2.0, 6.0, 1.0, 10.0],
                        ],
                        [
                            [2.0, 7.0, 1.0, 4.0],
                            [9.0, 3.0, 8.0, 2.0],
                            [1.0, 5.0, 2.0, 9.0],
                            [6.0, 2.0, 7.0, 3.0],
                        ],
                    ]
                ],
                requires_grad=True,
            )
            loss = (block(inputs) * spatial_weight).sum()
            loss.backward()
            with self.subTest(variant=variant):
                self.assertIsNotNone(inputs.grad)
                self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
                for parameter in block.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_context_term_is_supported_and_bounded_by_saliency(self) -> None:
        inputs = torch.randn(2, 3, 16, 20)
        for use_context_code in (False, True):
            block = TPDCleanV3Block(
                channels=3,
                activate=False,
                use_context_code=use_context_code,
            )
            _, context_term, saliency = block.fusion_terms(inputs)
            with self.subTest(use_context_code=use_context_code):
                self.assertTrue(
                    torch.all(context_term.abs() <= saliency.abs() + 1e-7)
                )
                zero_support = saliency == 0
                self.assertTrue(
                    torch.equal(
                        context_term.masked_select(zero_support),
                        torch.zeros_like(
                            context_term.masked_select(zero_support)
                        ),
                    )
                )

    def test_constant_context_has_zero_full_context_code(self) -> None:
        block = TPDCleanV3Block(
            channels=2,
            activate=False,
            use_context_code=True,
        )
        context = torch.full((1, 2, 4, 5), 3.25)
        saliency = torch.rand_like(context)
        self.assertTrue(
            torch.equal(
                block.context_code(context, saliency),
                torch.zeros_like(context),
            )
        )

    def test_capacity_control_uses_no_context_information(self) -> None:
        block = TPDCleanV3Block(
            channels=2,
            activate=False,
            use_context_code=False,
        )
        saliency = torch.rand(1, 2, 4, 5)
        code_a = block.context_code(torch.randn_like(saliency), saliency)
        code_b = block.context_code(torch.randn_like(saliency) * 100.0, saliency)
        self.assertTrue(torch.equal(code_a, torch.ones_like(saliency)))
        self.assertTrue(torch.equal(code_a, code_b))

    def test_parameter_counts_and_state_layout_are_capacity_matched(self) -> None:
        state_layouts = []
        for variant in SUPPORTED_CLEAN_V3_VARIANTS:
            emb1 = build_clean_v3_patch_embedding(variant, 32, 16)
            emb2 = build_clean_v3_patch_embedding(variant, 64, 8)
            with self.subTest(variant=variant):
                self.assertEqual(
                    parameter_count(emb1) + parameter_count(emb2), 66_496
                )
            state_layouts.append(
                (tuple(emb1.state_dict()), tuple(emb2.state_dict()))
            )
        self.assertTrue(
            all(layout == state_layouts[0] for layout in state_layouts[1:])
        )

    def test_strict_state_dict_round_trip(self) -> None:
        model = build_clean_v3_patch_embedding(
            PRIMARY_CLEAN_V3_VARIANT, channels=4, stride=8
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        rebuilt = build_clean_v3_patch_embedding(
            PRIMARY_CLEAN_V3_VARIANT, channels=4, stride=8
        )
        incompatible = rebuilt.load_state_dict(
            torch.load(buffer, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        inputs = torch.randn(2, 4, 32, 40)
        torch.testing.assert_close(
            rebuilt(inputs), model(inputs), rtol=0.0, atol=0.0
        )

    def test_none_and_invalid_input_contracts(self) -> None:
        embedding = build_clean_v3_patch_embedding(
            PRIMARY_CLEAN_V3_VARIANT, 4, 4
        )
        self.assertIsNone(embedding(None))
        with self.assertRaisesRegex(ValueError, "even H/W"):
            embedding(torch.randn(1, 4, 31, 32))
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v3"):
            clean_v3_variant_spec("unknown")


if __name__ == "__main__":
    unittest.main()
