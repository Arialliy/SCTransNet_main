from __future__ import annotations

import unittest

import torch

from experiments.train_tpd_pilot import weights_init_kaiming
from model.tpd import SPDPatchEmbedding
from model.tpd_clean import (
    CleanTPD2,
    CleanTPDPatchEmbedding,
    GroupedKeepPatchEmbedding,
    SUPPORTED_CLEAN_VARIANTS,
    build_clean_patch_embedding,
    parameter_count,
)


class TPDCleanTests(unittest.TestCase):
    def test_all_clean_variants_match_required_shapes(self) -> None:
        cases = ((32, 16, 192, 288), (64, 8, 96, 144))
        for channels, stride, height, width in cases:
            inputs = torch.randn(2, channels, height, width)
            expected = (2, channels, height // stride, width // stride)
            for variant in SUPPORTED_CLEAN_VARIANTS:
                with self.subTest(variant=variant, channels=channels):
                    output = build_clean_patch_embedding(
                        variant, channels, stride
                    )(inputs)
                    self.assertEqual(tuple(output.shape), expected)

    def test_zero_residual_scales_are_exactly_spd_equivalent(self) -> None:
        channels = 4
        stride = 8
        spd = SPDPatchEmbedding(channels, stride)
        clean = CleanTPDPatchEmbedding(
            channels,
            stride,
            use_context=True,
            use_saliency=True,
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
        clean = CleanTPDPatchEmbedding(
            channels,
            stride,
            use_context=True,
            use_saliency=True,
        )
        torch.manual_seed(29)
        spd.apply(weights_init_kaiming)
        torch.manual_seed(29)
        clean.apply(weights_init_kaiming)

        spd_x = torch.randn(2, channels, 32, 40)
        clean_x = spd_x.clone()
        for spd_block, clean_block in zip(spd.blocks, clean.blocks):
            self.assertTrue(torch.count_nonzero(clean_block.context_scale) == 0)
            self.assertTrue(torch.count_nonzero(clean_block.saliency_scale) == 0)
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

    def test_zero_initialized_scales_receive_finite_gradients(self) -> None:
        block = CleanTPD2(
            channels=1,
            activate=False,
            use_context=True,
            use_saliency=True,
        )
        with torch.no_grad():
            block.phase_compress.weight.fill_(0.25)
            block.phase_compress.bias.zero_()
        inputs = torch.tensor(
            [[[[1.0, 2.0], [3.0, 8.0]]]], requires_grad=True
        )
        block(inputs).sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        for name, parameter in block.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_odd_spatial_size_is_rejected_explicitly(self) -> None:
        block = CleanTPD2(
            channels=2,
            activate=False,
            use_context=True,
            use_saliency=True,
        )
        with self.assertRaisesRegex(ValueError, "even H/W"):
            block(torch.randn(1, 2, 7, 8))

    def test_parameter_counts_are_auditable(self) -> None:
        expected = {
            "tpd_clean_ctx": 66_176,
            "tpd_clean_sal": 66_176,
            "tpd_clean_full": 66_496,
            "grouped_keep": 1_280,
        }
        for variant, count in expected.items():
            emb1 = build_clean_patch_embedding(variant, 32, 16)
            emb2 = build_clean_patch_embedding(variant, 64, 8)
            with self.subTest(variant=variant):
                self.assertEqual(parameter_count(emb1) + parameter_count(emb2), count)
        self.assertEqual(expected["tpd_clean_full"] - 65_856, 640)

    def test_none_input_preserves_optional_embedding_contract(self) -> None:
        clean = CleanTPDPatchEmbedding(
            4, 4, use_context=True, use_saliency=True
        )
        grouped = GroupedKeepPatchEmbedding(4, 4)
        self.assertIsNone(clean(None))
        self.assertIsNone(grouped(None))


if __name__ == "__main__":
    unittest.main()
