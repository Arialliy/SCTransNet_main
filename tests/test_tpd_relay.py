from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from experiments.train_tpd_pilot import deep_supervision_loss, weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd import build_patch_embedding
from model.tpd_clean import CleanTPDPatchEmbedding, build_clean_patch_embedding
from model.tpd_relay import (
    NestedEvidenceRelay,
    RelayRuntime,
    TPDEvidenceTap,
    install_tpd_ner,
    relay_parameter_count,
)


def build_small_clean_model(*, deepsuper: bool) -> tuple[SCTransNet, dict]:
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    model = SCTransNet(
        config,
        img_size=32,
        mode="train",
        deepsuper=deepsuper,
    )
    torch.manual_seed(31)
    model.apply(weights_init_kaiming)
    replacements = {
        "embeddings_1": build_clean_patch_embedding(
            "tpd_clean_full",
            channels=4,
            stride=16,
        ),
        "embeddings_2": build_clean_patch_embedding(
            "tpd_clean_full",
            channels=8,
            stride=8,
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    torch.manual_seed(47)
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model, replacements


class TPDRelayTests(unittest.TestCase):
    def test_evidence_tap_preserves_output_and_captures_dynamic_shapes(self) -> None:
        embedding = CleanTPDPatchEmbedding(
            channels=4,
            stride=16,
            use_context=True,
            use_saliency=True,
        )
        inputs = torch.randn(2, 4, 64, 80)
        expected = embedding(inputs)
        runtime = RelayRuntime()
        tap = TPDEvidenceTap(
            embedding,
            "embeddings_1",
            runtime,
            expected_blocks=4,
        )
        actual = tap(inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(
            [tuple(value.shape) for value in runtime.evidence("embeddings_1")],
            [
                (2, 4, 32, 40),
                (2, 4, 16, 20),
                (2, 4, 8, 10),
            ],
        )

    def test_relay_parameter_budget_is_exact(self) -> None:
        relay = NestedEvidenceRelay(
            RelayRuntime(),
            base_channels=32,
            width=8,
        )
        self.assertEqual(relay_parameter_count(relay), 11_291)

    def test_zero_gates_make_full_model_exactly_equivalent(self) -> None:
        baseline, _ = build_small_clean_model(deepsuper=True)
        candidate = copy.deepcopy(baseline)
        replacements = {
            "embeddings_1": candidate.mtc.embeddings_1,
            "embeddings_2": candidate.mtc.embeddings_2,
        }
        parts = install_tpd_ner(candidate, replacements, width=2)
        baseline.eval()
        candidate.eval()
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            expected = baseline(inputs)
            actual = candidate(inputs)
        self.assertEqual(len(expected), 6)
        self.assertEqual(len(actual), 6)
        for expected_output, actual_output in zip(expected, actual):
            self.assertTrue(torch.equal(expected_output, actual_output))
            self.assertEqual(tuple(actual_output.shape), (2, 1, 32, 32))
        self.assertEqual(
            parts["relay"].runtime.shape_snapshot()["q4"],
            (2, 2, 4, 4),
        )
        self.assertEqual(
            parts["relay"].runtime.shape_snapshot()["q3"],
            (2, 2, 8, 8),
        )
        self.assertEqual(
            parts["relay"].runtime.shape_snapshot()["q2"],
            (2, 2, 16, 16),
        )

    def test_complete_model_supports_repeated_dynamic_forward_sizes(self) -> None:
        model, replacements = build_small_clean_model(deepsuper=True)
        relay = install_tpd_ner(model, replacements, width=2)["relay"]
        model.eval()

        for height, width in ((32, 32), (64, 96), (32, 64)):
            with self.subTest(size=(height, width)):
                inputs = torch.randn(2, 1, height, width)
                with torch.no_grad():
                    outputs = model(inputs)
                self.assertEqual(len(outputs), 6)
                for output in outputs:
                    self.assertEqual(tuple(output.shape), (2, 1, height, width))
                    self.assertTrue(torch.isfinite(output).all())
                shapes = relay.runtime.shape_snapshot()
                self.assertEqual(shapes["q4"], (2, 2, height // 8, width // 8))
                self.assertEqual(shapes["q3"], (2, 2, height // 4, width // 4))
                self.assertEqual(shapes["q2"], (2, 2, height // 2, width // 2))
                self.assertFalse(relay.runtime._active)
                self.assertFalse(relay.runtime._evidence)
                self.assertFalse(relay.runtime._relay)

    def test_zero_gates_receive_gradient_on_first_backward(self) -> None:
        model, replacements = build_small_clean_model(deepsuper=False)
        relay = install_tpd_ner(model, replacements, width=2)["relay"]
        model.eval()
        inputs = torch.randn(2, 1, 32, 32)
        model(inputs).mean().backward()
        for stage, gate in relay.gates.items():
            with self.subTest(stage=stage):
                self.assertIsNotNone(gate.weight.grad)
                self.assertTrue(torch.isfinite(gate.weight.grad).all())
                self.assertGreater(float(gate.weight.grad.abs().sum()), 0.0)

    def test_installed_model_state_dict_loads_strictly(self) -> None:
        source, source_replacements = build_small_clean_model(deepsuper=False)
        install_tpd_ner(source, source_replacements, width=2)
        state = source.state_dict()
        relay_keys = [key for key in state if key.startswith("tpd_ner.")]
        self.assertTrue(relay_keys)
        self.assertFalse(
            any("binding" in key or ".relay." in key for key in state)
        )
        parameter_ids = [
            id(parameter)
            for _, parameter in source.named_parameters(remove_duplicate=False)
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

        target, target_replacements = build_small_clean_model(deepsuper=False)
        install_tpd_ner(target, target_replacements, width=2)
        target.load_state_dict(state, strict=True)
        self.assertEqual(set(state), set(target.state_dict()))

    def test_install_rejects_mismatched_replacement_contract(self) -> None:
        model, _ = build_small_clean_model(deepsuper=False)
        with self.assertRaisesRegex(ValueError, "does not match"):
            install_tpd_ner(
                model,
                {
                    "embeddings_1": CleanTPDPatchEmbedding(
                        4,
                        16,
                        use_context=True,
                        use_saliency=True,
                    ),
                    "embeddings_2": model.mtc.embeddings_2,
                },
                width=2,
            )

    def test_install_rejects_all_non_full_tpd_embeddings(self) -> None:
        cases = {
            "spd": lambda channels, stride: build_patch_embedding(
                "spd", channels, stride
            ),
            "grouped_keep": lambda channels, stride: build_clean_patch_embedding(
                "grouped_keep", channels, stride
            ),
            "context_only": lambda channels, stride: build_clean_patch_embedding(
                "tpd_clean_ctx", channels, stride
            ),
            "saliency_only": lambda channels, stride: build_clean_patch_embedding(
                "tpd_clean_sal", channels, stride
            ),
            "tpd_v1": lambda channels, stride: build_patch_embedding(
                "tpd", channels, stride
            ),
        }
        for label, factory in cases.items():
            model, _ = build_small_clean_model(deepsuper=False)
            replacements = {
                "embeddings_1": factory(4, 16),
                "embeddings_2": factory(8, 8),
            }
            model.mtc.embeddings_1 = replacements["embeddings_1"]
            model.mtc.embeddings_2 = replacements["embeddings_2"]
            with self.subTest(candidate=label):
                with self.assertRaisesRegex(
                    TypeError,
                    "CleanTPDPatchEmbedding|full Keep-Context-Saliency",
                ):
                    install_tpd_ner(model, replacements, width=2)
                self.assertFalse(hasattr(model, "tpd_ner"))

    def test_installed_model_deepcopy_has_independent_relay_binding(self) -> None:
        source, replacements = build_small_clean_model(deepsuper=False)
        install_tpd_ner(source, replacements, width=2)
        candidate = copy.deepcopy(source)

        self.assertIs(candidate.up_decoder4.relay, candidate.tpd_ner)
        self.assertIs(candidate.up_decoder3.relay, candidate.tpd_ner)
        self.assertIs(candidate.up_decoder2.relay, candidate.tpd_ner)
        self.assertIs(candidate.mtc.embeddings_1.runtime, candidate.tpd_ner.runtime)
        self.assertIs(candidate.mtc.embeddings_2.runtime, candidate.tpd_ner.runtime)
        self.assertIsNot(candidate.tpd_ner, source.tpd_ner)
        self.assertIsNot(candidate.tpd_ner.runtime, source.tpd_ner.runtime)

        source.eval()
        candidate.eval()
        inputs = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            expected = source(inputs)
            actual = candidate(inputs)
        self.assertTrue(torch.equal(expected, actual))

    def test_failed_decoder_validation_leaves_model_uninstalled(self) -> None:
        model, replacements = build_small_clean_model(deepsuper=False)
        embedding1 = model.mtc.embeddings_1
        embedding2 = model.mtc.embeddings_2
        decoder3 = model.up_decoder3
        decoder2 = model.up_decoder2
        incompatible_decoder = nn.Identity()
        model.up_decoder4 = incompatible_decoder

        with self.assertRaisesRegex(TypeError, "compatible CCA decoder"):
            install_tpd_ner(model, replacements, width=2)
        self.assertIs(model.mtc.embeddings_1, embedding1)
        self.assertIs(model.mtc.embeddings_2, embedding2)
        self.assertIs(model.up_decoder4, incompatible_decoder)
        self.assertIs(model.up_decoder3, decoder3)
        self.assertIs(model.up_decoder2, decoder2)
        self.assertFalse(hasattr(model, "tpd_ner"))

    def test_relay_fusion_learns_after_zero_gate_first_step(self) -> None:
        model, replacements = build_small_clean_model(deepsuper=True)
        relay = install_tpd_ner(model, replacements, width=2)["relay"]
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn(2, 1, 32, 32)
        targets = torch.rand(2, 1, 32, 32)
        criterion = nn.BCELoss(reduction="mean")
        tpd_scales = [
            scale
            for embedding in replacements.values()
            for block in embedding.blocks
            for scale in (block.context_scale, block.saliency_scale)
        ]
        scale_before = [scale.detach().clone() for scale in tpd_scales]
        relay_parameter_ids = {id(parameter) for parameter in relay.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(relay_parameter_ids <= optimizer_parameter_ids)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
        loss = deep_supervision_loss(outputs, targets, criterion)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertTrue(
            any(
                not torch.equal(before, scale.detach())
                for before, scale in zip(scale_before, tpd_scales)
            )
        )
        self.assertGreater(
            sum(
                float(gate.weight.detach().abs().sum())
                for gate in relay.gates.values()
            ),
            0.0,
        )

        optimizer.zero_grad(set_to_none=True)
        second_outputs = model(inputs)
        second_loss = deep_supervision_loss(second_outputs, targets, criterion)
        self.assertTrue(torch.isfinite(second_loss))
        second_loss.backward()
        fusion_gradients = {
            stage: sum(
                float(parameter.grad.abs().sum())
                for parameter in fusion.parameters()
                if parameter.grad is not None
            )
            for stage, fusion in relay.fusions.items()
        }
        for stage, gradient in fusion_gradients.items():
            with self.subTest(stage=stage):
                self.assertGreater(gradient, 0.0)
        fusion_before = {
            name: parameter.detach().clone()
            for name, parameter in relay.fusions.named_parameters()
        }
        optimizer.step()
        for stage in relay.fusions:
            with self.subTest(stage=stage):
                prefix = f"{stage}."
                self.assertTrue(
                    any(
                        not torch.equal(fusion_before[name], parameter.detach())
                        for name, parameter in relay.fusions.named_parameters()
                        if name.startswith(prefix)
                    )
                )


if __name__ == "__main__":
    unittest.main()
