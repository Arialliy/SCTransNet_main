from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from experiments.train_tpd_pilot import deep_supervision_loss, weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean import build_clean_patch_embedding
from model.tpd_relay import install_tpd_ner
from model.tpd_sctransnet import (
    EVIDENCE_NODE_NAMES,
    RELAY_STAGE_ORDER,
    ExplicitTPDEvidenceEmbedding,
    TPDSCTransNet,
)


def small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def build_legacy_runtime_model() -> SCTransNet:
    model = SCTransNet(
        small_config(),
        img_size=32,
        mode="train",
        deepsuper=True,
    )
    torch.manual_seed(31)
    model.apply(weights_init_kaiming)
    replacements = {
        "embeddings_1": build_clean_patch_embedding(
            "tpd_clean_full", channels=4, stride=16
        ),
        "embeddings_2": build_clean_patch_embedding(
            "tpd_clean_full", channels=8, stride=8
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    torch.manual_seed(47)
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    relay = install_tpd_ner(model, replacements, width=2)["relay"]
    relay.apply(weights_init_kaiming)
    relay.zero_init_gates()
    return model


class FirstClassTPDSCTransNetTests(unittest.TestCase):
    def test_manifest_records_three_branch_core_and_five_evidence_nodes(self) -> None:
        model = TPDSCTransNet(
            small_config(),
            img_size=32,
            relay_width=2,
        )
        manifest = model.architecture_manifest()
        self.assertEqual(manifest["model"], "TPDSCTransNet")
        self.assertEqual(
            manifest["primary_module"],
            "Keep-Context-Saliency TPD",
        )
        self.assertEqual(manifest["evidence_nodes"], EVIDENCE_NODE_NAMES)
        self.assertEqual(manifest["relay_stage_order"], RELAY_STAGE_ORDER)
        self.assertEqual(manifest["tensor_handoff"], "forward_local_explicit")
        self.assertFalse(hasattr(model.tpd_ner, "runtime"))
        self.assertFalse(hasattr(model.up_decoder4, "binding"))

    def test_explicit_embedding_returns_exact_five_dynamic_nodes(self) -> None:
        model = TPDSCTransNet(
            small_config(),
            img_size=32,
            relay_width=2,
        )
        x1 = torch.randn(2, 4, 64, 96)
        x2 = torch.randn(2, 8, 32, 48)
        emb1, nodes1 = model.mtc.embeddings_1.forward_with_evidence(x1)
        emb2, nodes2 = model.mtc.embeddings_2.forward_with_evidence(x2)
        self.assertIsInstance(model.mtc.embeddings_1, ExplicitTPDEvidenceEmbedding)
        self.assertEqual(
            [tuple(node.shape) for node in nodes1],
            [(2, 4, 32, 48), (2, 4, 16, 24), (2, 4, 8, 12)],
        )
        self.assertEqual(
            [tuple(node.shape) for node in nodes2],
            [(2, 8, 16, 24), (2, 8, 8, 12)],
        )
        self.assertEqual(tuple(emb1.shape), (2, 4, 4, 6))
        self.assertEqual(tuple(emb2.shape), (2, 8, 4, 6))

    def test_legacy_state_dict_loads_strictly_and_outputs_are_exact(self) -> None:
        legacy = build_legacy_runtime_model()
        explicit = TPDSCTransNet(
            small_config(),
            img_size=32,
            mode="train",
            deepsuper=True,
            relay_width=2,
        )
        self.assertEqual(set(legacy.state_dict()), set(explicit.state_dict()))
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        legacy.eval()
        explicit.eval()
        for size in ((32, 32), (64, 96)):
            with self.subTest(size=size):
                inputs = torch.randn(2, 1, *size)
                with torch.no_grad():
                    expected = legacy(inputs)
                    actual = explicit(inputs)
                self.assertEqual(len(expected), 6)
                self.assertEqual(len(actual), 6)
                for expected_output, actual_output in zip(expected, actual):
                    self.assertTrue(torch.equal(expected_output, actual_output))

    def test_nonzero_legacy_relay_matches_explicit_forward_and_backward(self) -> None:
        torch.manual_seed(701)
        legacy = build_legacy_runtime_model()
        explicit = TPDSCTransNet(
            small_config(),
            img_size=32,
            mode="train",
            deepsuper=True,
            relay_width=2,
        )
        legacy.eval()
        explicit.eval()
        torch.manual_seed(702)
        inputs = torch.randn(2, 1, 32, 64)
        targets = torch.rand(2, 1, 32, 64)

        with torch.no_grad():
            zero_gate_outputs = tuple(output.clone() for output in legacy(inputs))
            for stage, gate in legacy.tpd_ner.gates.items():
                gate.weight.fill_(0.025 * int(stage))
                if gate.bias is not None:
                    gate.bias.zero_()
                self.assertEqual(
                    int(torch.count_nonzero(gate.weight)),
                    gate.weight.numel(),
                )

        self.assertEqual(set(legacy.state_dict()), set(explicit.state_dict()))
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        legacy_parameters = dict(legacy.named_parameters())
        explicit_parameters = dict(explicit.named_parameters())
        self.assertEqual(list(legacy_parameters), list(explicit_parameters))

        expected = legacy(inputs)
        actual = explicit(inputs)
        self.assertEqual(len(expected), 6)
        self.assertEqual(len(actual), 6)
        self.assertFalse(torch.equal(zero_gate_outputs[-1], expected[-1]))
        for expected_output, actual_output in zip(expected, actual):
            self.assertEqual(tuple(expected_output.shape), (2, 1, 32, 64))
            self.assertTrue(torch.equal(expected_output, actual_output))

        criterion = nn.BCELoss(reduction="mean")
        expected_loss = deep_supervision_loss(expected, targets, criterion)
        actual_loss = deep_supervision_loss(actual, targets, criterion)
        self.assertTrue(torch.equal(expected_loss, actual_loss))
        expected_loss.backward()
        actual_loss.backward()

        for name in legacy_parameters:
            expected_gradient = legacy_parameters[name].grad
            actual_gradient = explicit_parameters[name].grad
            with self.subTest(parameter=name):
                self.assertEqual(
                    expected_gradient is None,
                    actual_gradient is None,
                )
                if expected_gradient is not None:
                    self.assertTrue(torch.equal(expected_gradient, actual_gradient))

    def test_two_steps_train_gates_then_relay_fusions(self) -> None:
        model = TPDSCTransNet(
            small_config(),
            img_size=32,
            mode="train",
            deepsuper=True,
            relay_width=2,
        )
        model.apply(weights_init_kaiming)
        model.zero_init_target_residuals()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss(reduction="mean")
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.rand(2, 1, 32, 32)

        optimizer.zero_grad(set_to_none=True)
        loss = deep_supervision_loss(model(inputs), targets, criterion)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for stage, gate in model.tpd_ner.gates.items():
            with self.subTest(step=1, stage=stage):
                self.assertIsNotNone(gate.weight.grad)
                self.assertGreater(float(gate.weight.grad.abs().sum()), 0.0)
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

    def test_phased_installation_blocks_incomplete_forward(self) -> None:
        model = TPDSCTransNet(
            small_config(),
            img_size=32,
            relay_width=2,
            install_tpd=False,
        )
        inputs = torch.randn(2, 1, 32, 32)
        with self.assertRaisesRegex(RuntimeError, "installation incomplete"):
            model(inputs)
        replacements = model.install_tpd_tokenizer()
        self.assertEqual(set(replacements), {"embeddings_1", "embeddings_2"})
        with self.assertRaisesRegex(RuntimeError, "installation incomplete"):
            model(inputs)
        parts = model.install_nested_relay()
        self.assertEqual(set(parts), {
            "embedding_1",
            "embedding_2",
            "relay",
            "decoder_4",
            "decoder_3",
            "decoder_2",
        })
        outputs = model(inputs)
        self.assertEqual(len(outputs), 6)


if __name__ == "__main__":
    unittest.main()
