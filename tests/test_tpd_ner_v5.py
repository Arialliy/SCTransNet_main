from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from experiments.train_tpd_pilot import (
    deep_supervision_loss,
    weights_init_kaiming,
)
from model.Config import get_SCTrans_config
from model.tpd_clean_v5 import (
    PRIMARY_CLEAN_V5_VARIANT,
    TPDCleanV5Block,
)
from model.tpd_ner_v5 import (
    CapacityMatchedProgressiveBlock,
    CapacityMatchedProgressivePatchEmbedding,
    EVIDENCE_NODE_NAMES,
    PROGRESSIVE_TOKENIZER,
    RELAY_STAGE_ORDER,
    ExplicitV5EvidenceEmbedding,
    TPDNERV5SCTransNet,
)


torch.set_num_threads(1)


def small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def build_small_model(
    tokenizer_variant: str,
    *,
    relay_enabled: bool,
) -> TPDNERV5SCTransNet:
    torch.manual_seed(101)
    model = TPDNERV5SCTransNet(
        small_config(),
        img_size=32,
        mode="train",
        deepsuper=True,
        tokenizer_variant=tokenizer_variant,
        relay_enabled=relay_enabled,
        relay_width=2,
    )
    model.apply(weights_init_kaiming)
    model.zero_init_extension_residuals()
    return model


class TPDNERV5ModelTests(unittest.TestCase):
    def test_v5_preserves_three_sources_one_scale_and_five_nodes(self) -> None:
        model = build_small_model(
            PRIMARY_CLEAN_V5_VARIANT,
            relay_enabled=False,
        )
        embedding1 = model.mtc.embeddings_1
        embedding2 = model.mtc.embeddings_2
        self.assertIsInstance(embedding1, ExplicitV5EvidenceEmbedding)
        self.assertIsInstance(embedding2, ExplicitV5EvidenceEmbedding)

        blocks = tuple(embedding1.blocks) + tuple(embedding2.blocks)
        self.assertEqual(len(blocks), 7)
        self.assertTrue(all(isinstance(block, TPDCleanV5Block) for block in blocks))
        self.assertTrue(all(block.use_context_selector for block in blocks))
        self.assertTrue(all(not hasattr(block, "context_scale") for block in blocks))
        self.assertEqual(
            sum(
                name.endswith("saliency_scale")
                for name, _ in model.named_parameters()
            ),
            7,
        )

        input1 = torch.randn(2, 4, 64, 96)
        input2 = torch.randn(2, 8, 32, 48)
        endpoint1, nodes1 = embedding1.forward_with_evidence(input1)
        endpoint2, nodes2 = embedding2.forward_with_evidence(input2)
        self.assertEqual(
            [tuple(node.shape) for node in nodes1],
            [(2, 4, 32, 48), (2, 4, 16, 24), (2, 4, 8, 12)],
        )
        self.assertEqual(
            [tuple(node.shape) for node in nodes2],
            [(2, 8, 16, 24), (2, 8, 8, 12)],
        )
        self.assertEqual(tuple(endpoint1.shape), (2, 4, 4, 6))
        self.assertEqual(tuple(endpoint2.shape), (2, 8, 4, 6))

        keep, context, saliency = embedding1.blocks[0].branches(input1)
        self.assertEqual(keep.shape, context.shape)
        self.assertEqual(keep.shape, saliency.shape)
        manifest = model.architecture_manifest()
        self.assertEqual(
            manifest["semantic_sources"],
            ("Keep", "Context", "Saliency"),
        )
        self.assertEqual(manifest["evidence_nodes"], EVIDENCE_NODE_NAMES)
        self.assertFalse(manifest["relay_enabled"])
        self.assertFalse(manifest["fourth_parallel_branch_added"])
        self.assertFalse(hasattr(model, "tpd_ner"))

    def test_progressive_control_exposes_same_five_node_geometry(self) -> None:
        model = build_small_model(
            PROGRESSIVE_TOKENIZER,
            relay_enabled=False,
        )
        embedding1 = model.mtc.embeddings_1
        embedding2 = model.mtc.embeddings_2
        input1 = torch.randn(2, 4, 64, 96)
        input2 = torch.randn(2, 8, 32, 48)
        endpoint1, nodes1 = embedding1.forward_with_evidence(input1)
        endpoint2, nodes2 = embedding2.forward_with_evidence(input2)
        self.assertEqual(
            [tuple(node.shape) for node in nodes1],
            [(2, 4, 32, 48), (2, 4, 16, 24), (2, 4, 8, 12)],
        )
        self.assertEqual(
            [tuple(node.shape) for node in nodes2],
            [(2, 8, 16, 24), (2, 8, 8, 12)],
        )
        self.assertEqual(tuple(endpoint1.shape), (2, 4, 4, 6))
        self.assertEqual(tuple(endpoint2.shape), (2, 8, 4, 6))
        self.assertFalse(
            any("saliency_scale" in name for name, _ in model.named_parameters())
        )
        blocks = tuple(embedding1.blocks) + tuple(embedding2.blocks)
        self.assertTrue(
            all(
                isinstance(block, CapacityMatchedProgressiveBlock)
                for block in blocks
            )
        )
        self.assertTrue(
            all(int(torch.count_nonzero(block.channel_gain)) == 0 for block in blocks)
        )
        for block in blocks:
            expected = 4 * block.channels * block.channels + 2 * block.channels
            actual = sum(parameter.numel() for parameter in block.parameters())
            self.assertEqual(actual, expected)

    def test_progressive_shallow_capacity_is_exactly_v5_66176(self) -> None:
        embedding1 = CapacityMatchedProgressivePatchEmbedding(32, 16)
        embedding2 = CapacityMatchedProgressivePatchEmbedding(64, 8)
        parameters = sum(
            parameter.numel()
            for embedding in (embedding1, embedding2)
            for parameter in embedding.parameters()
        )
        self.assertEqual(len(embedding1.blocks), 4)
        self.assertEqual(len(embedding2.blocks), 3)
        self.assertEqual(parameters, 66_176)

    def test_every_progressive_capacity_parameter_has_gradient(self) -> None:
        model = build_small_model(
            PROGRESSIVE_TOKENIZER,
            relay_enabled=False,
        )
        model.train()
        inputs = torch.randn(2, 1, 32, 64)
        targets = torch.rand(2, 1, 32, 64)
        loss = deep_supervision_loss(
            model(inputs),
            targets,
            nn.BCELoss(reduction="mean"),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        shallow_parameters = []
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(model.mtc, embedding_name)
            shallow_parameters.extend(
                (f"{embedding_name}.{name}", parameter)
                for name, parameter in embedding.named_parameters()
            )
        self.assertEqual(len(shallow_parameters), 21)
        for name, parameter in shallow_parameters:
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_relay_uses_explicit_dynamic_q4_q3_q2_forward(self) -> None:
        model = build_small_model(
            PRIMARY_CLEAN_V5_VARIANT,
            relay_enabled=True,
        )
        manifest = model.architecture_manifest()
        self.assertEqual(manifest["evidence_nodes"], EVIDENCE_NODE_NAMES)
        self.assertEqual(manifest["relay_stage_order"], RELAY_STAGE_ORDER)
        self.assertEqual(manifest["tensor_handoff"], "forward_local_explicit")
        self.assertTrue(manifest["relay_enabled"])
        self.assertFalse(hasattr(model.tpd_ner, "runtime"))
        self.assertTrue(
            all(
                name.startswith("tpd_ner.")
                for name in model.state_dict()
                if name.startswith("tpd_ner.")
            )
        )
        for gate in model.tpd_ner.gates.values():
            self.assertEqual(int(torch.count_nonzero(gate.weight)), 0)
            self.assertEqual(int(torch.count_nonzero(gate.bias)), 0)

        model.eval()
        for height, width in ((32, 32), (32, 64), (64, 96)):
            with self.subTest(size=(height, width)):
                inputs = torch.randn(2, 1, height, width)
                with torch.no_grad():
                    outputs = model(inputs)
                self.assertEqual(len(outputs), 6)
                for output in outputs:
                    self.assertEqual(
                        tuple(output.shape),
                        (2, 1, height, width),
                    )
                    self.assertTrue(torch.isfinite(output).all())

    def test_two_steps_train_zero_gates_then_relay_fusions(self) -> None:
        model = build_small_model(
            PRIMARY_CLEAN_V5_VARIANT,
            relay_enabled=True,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss(reduction="mean")
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.rand(2, 1, 32, 32)

        optimizer.zero_grad(set_to_none=True)
        first_loss = deep_supervision_loss(model(inputs), targets, criterion)
        self.assertTrue(torch.isfinite(first_loss))
        first_loss.backward()
        for stage, gate in model.tpd_ner.gates.items():
            with self.subTest(step=1, stage=stage):
                self.assertIsNotNone(gate.weight.grad)
                self.assertGreater(float(gate.weight.grad.abs().sum()), 0.0)
        scale_gradient = sum(
            float(block.saliency_scale.grad.abs().sum())
            for name in ("embeddings_1", "embeddings_2")
            for block in getattr(model.mtc, name).blocks
        )
        self.assertGreater(scale_gradient, 0.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        second_loss = deep_supervision_loss(model(inputs), targets, criterion)
        self.assertTrue(torch.isfinite(second_loss))
        second_loss.backward()
        for stage, fusion in model.tpd_ner.fusions.items():
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in fusion.parameters()
                if parameter.grad is not None
            )
            with self.subTest(step=2, stage=stage):
                self.assertGreater(gradient, 0.0)
        for name, parameter in model.tpd_ner.named_parameters():
            with self.subTest(step=2, relay_parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
