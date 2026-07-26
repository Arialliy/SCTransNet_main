from __future__ import annotations

import unittest

import torch

from model.tpd_clean_v3 import TPDCleanV3Block, TPDCleanV3PatchEmbedding


class TPDCleanV3StructureContractTests(unittest.TestCase):
    def test_dense_keep_and_exact_learned_state_contract(self) -> None:
        block = TPDCleanV3Block(
            channels=4,
            activate=False,
            use_context_code=True,
        )

        self.assertEqual(block.phase_compress.in_channels, 16)
        self.assertEqual(block.phase_compress.out_channels, 4)
        self.assertEqual(block.phase_compress.kernel_size, (1, 1))
        self.assertEqual(block.phase_compress.groups, 1)
        self.assertEqual(
            set(dict(block.named_children())),
            {"phase_compress", "activation"},
        )
        self.assertEqual(
            set(dict(block.named_parameters())),
            {
                "context_scale",
                "saliency_scale",
                "phase_compress.weight",
                "phase_compress.bias",
            },
        )

    def test_single_impulse_is_phase_aligned_in_exactly_three_sources(self) -> None:
        block = TPDCleanV3Block(
            channels=1,
            activate=False,
            use_context_code=True,
        )
        with torch.no_grad():
            block.phase_compress.weight.zero_()
            block.phase_compress.bias.zero_()
            block.phase_compress.weight[0, 0, 0, 0] = 1.0

        inputs = torch.zeros(1, 1, 4, 4)
        inputs[0, 0, 2, 2] = 4.0
        branches = block.branches(inputs)

        self.assertEqual(len(branches), 3)
        keep, context, saliency = branches
        self.assertEqual(tuple(keep.shape), (1, 1, 2, 2))
        self.assertTrue(torch.equal(keep.nonzero(), torch.tensor([[0, 0, 1, 1]])))
        self.assertTrue(
            torch.equal(context.nonzero(), torch.tensor([[0, 0, 1, 1]]))
        )
        self.assertTrue(
            torch.equal(saliency.nonzero(), torch.tensor([[0, 0, 1, 1]]))
        )
        self.assertEqual(keep[0, 0, 1, 1].detach().item(), 4.0)
        self.assertEqual(context[0, 0, 1, 1].item(), 1.0)
        self.assertEqual(saliency[0, 0, 1, 1].item(), 3.0)

    def test_hierarchical_embedding_is_a_serial_chain_of_kcs_blocks(self) -> None:
        embedding = TPDCleanV3PatchEmbedding(
            channels=2,
            stride=8,
            use_context_code=True,
        )

        self.assertEqual(len(embedding.blocks), 3)
        self.assertTrue(
            all(isinstance(block, TPDCleanV3Block) for block in embedding.blocks)
        )
        self.assertTrue(
            all(block.phase_compress.groups == 1 for block in embedding.blocks)
        )


if __name__ == "__main__":
    unittest.main()
