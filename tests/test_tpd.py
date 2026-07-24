from __future__ import annotations

import unittest

import torch

from model.tpd import (
    ProgressivePatchEmbedding,
    SPDPatchEmbedding,
    TPD2,
    TPDPatchEmbedding,
    parameter_count,
)


class TPDModuleTests(unittest.TestCase):
    def test_all_variants_match_required_shapes(self) -> None:
        cases = ((32, 16, 192, 288), (64, 8, 96, 144))
        classes = (ProgressivePatchEmbedding, SPDPatchEmbedding, TPDPatchEmbedding)
        for channels, stride, height, width in cases:
            inputs = torch.randn(2, channels, height, width)
            expected = (2, channels, height // stride, width // stride)
            for module_class in classes:
                with self.subTest(module=module_class.__name__, channels=channels):
                    output = module_class(channels, stride)(inputs)
                    self.assertEqual(tuple(output.shape), expected)

    def test_tpd_branches_use_the_same_two_by_two_lattice(self) -> None:
        block = TPD2(channels=1, activate=False)
        with torch.no_grad():
            block.phase_compress.weight.fill_(0.0)
            # Select the top-left phase from PixelUnshuffle's four phases.
            block.phase_compress.weight[0, 0, 0, 0] = 1.0
        inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        keep, context, saliency = block.branches(inputs)
        self.assertEqual(float(keep.item()), 1.0)
        self.assertEqual(float(context.item()), 2.5)
        self.assertEqual(float(saliency.item()), 1.5)

    def test_tpd_gradients_reach_input_and_learned_paths(self) -> None:
        module = TPDPatchEmbedding(channels=4, stride=8)
        inputs = torch.randn(2, 4, 32, 40, requires_grad=True)
        module(inputs).square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_odd_spatial_size_is_rejected_explicitly(self) -> None:
        block = TPD2(channels=2, activate=False)
        with self.assertRaisesRegex(ValueError, "even H/W"):
            block(torch.randn(1, 2, 7, 8))

    def test_parameter_counts_are_auditable(self) -> None:
        emb1 = TPDPatchEmbedding(32, 16)
        emb2 = TPDPatchEmbedding(64, 8)
        self.assertEqual(parameter_count(emb1), 12_928)
        self.assertEqual(parameter_count(emb2), 37_824)
        self.assertEqual(parameter_count(emb1) + parameter_count(emb2), 50_752)

    def test_spd_control_uses_dense_space_to_depth_projection(self) -> None:
        emb1 = SPDPatchEmbedding(32, 16)
        emb2 = SPDPatchEmbedding(64, 8)
        self.assertEqual(parameter_count(emb1) + parameter_count(emb2), 65_856)


if __name__ == "__main__":
    unittest.main()
