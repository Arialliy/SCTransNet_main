from __future__ import annotations

import io
import unittest

import torch

from experiments.train_tpd_pilot import weights_init_kaiming
from model.tpd import SPDPatchEmbedding
from model.tpd_clean_v5 import (
    CONTEXT_SELECTOR_CEILING,
    CONTEXT_SELECTOR_FLOOR,
    PRIMARY_CLEAN_V5_VARIANT,
    SUPPORTED_CLEAN_V5_VARIANTS,
    TPDCleanV5Block,
    build_clean_v5_patch_embedding,
    clean_v5_variant_spec,
    parameter_count,
)


class TPDCleanV5Tests(unittest.TestCase):
    def test_candidate_matrix_keeps_only_kcs_and_one_scale(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V5_VARIANTS,
            (
                "tpd_clean_v5_full",
                "tpd_clean_v5_sal_capacity",
            ),
        )
        self.assertEqual(PRIMARY_CLEAN_V5_VARIANT, "tpd_clean_v5_full")
        full = clean_v5_variant_spec("tpd_clean_v5_full")
        control = clean_v5_variant_spec("tpd_clean_v5_sal_capacity")
        self.assertEqual(
            full["context_selector"],
            "positive_centered_0p5_to_1p5",
        )
        self.assertEqual(control["context_selector"], "neutral_one")
        self.assertEqual(
            full["fusion_support"],
            "positive_context_selected_saliency",
        )
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])

        block = TPDCleanV5Block(
            3,
            activate=False,
            use_context_selector=True,
        )
        self.assertEqual(
            set(dict(block.named_parameters())),
            {
                "saliency_scale",
                "phase_compress.weight",
                "phase_compress.bias",
            },
        )
        self.assertFalse(hasattr(block, "context_scale"))
        self.assertEqual(
            set(dict(block.named_children())),
            {"phase_compress", "activation"},
        )
        self.assertEqual(block.phase_compress.groups, 1)
        self.assertEqual(block.phase_compress.kernel_size, (1, 1))
        self.assertEqual(block.phase_compress.in_channels, 12)
        self.assertEqual(block.phase_compress.out_channels, 3)

    def test_dense_keep_context_saliency_sources_and_shapes(self) -> None:
        block = TPDCleanV5Block(
            1,
            activate=False,
            use_context_selector=True,
        )
        with torch.no_grad():
            block.phase_compress.weight.zero_()
            block.phase_compress.bias.zero_()
            block.phase_compress.weight[0, 0, 0, 0] = 1.0
        inputs = torch.zeros(1, 1, 4, 4)
        inputs[0, 0, 2, 2] = 4.0
        keep, context, saliency = block.branches(inputs)
        location = torch.tensor([[0, 0, 1, 1]])
        self.assertTrue(torch.equal(keep.nonzero(), location))
        self.assertTrue(torch.equal(context.nonzero(), location))
        self.assertTrue(torch.equal(saliency.nonzero(), location))
        self.assertEqual(float(keep[0, 0, 1, 1].detach()), 4.0)
        self.assertEqual(float(context[0, 0, 1, 1]), 1.0)
        self.assertEqual(float(saliency[0, 0, 1, 1]), 3.0)

        cases = (
            (32, 16, 64, 96, 4),
            (64, 8, 48, 72, 3),
        )
        for channels, stride, height, width, expected_blocks in cases:
            candidate_inputs = torch.randn(1, channels, height, width)
            for variant in SUPPORTED_CLEAN_V5_VARIANTS:
                embedding = build_clean_v5_patch_embedding(
                    variant,
                    channels,
                    stride,
                )
                with self.subTest(variant=variant, channels=channels):
                    self.assertEqual(len(embedding.blocks), expected_blocks)
                    self.assertEqual(
                        tuple(embedding(candidate_inputs).shape),
                        (
                            1,
                            channels,
                            height // stride,
                            width // stride,
                        ),
                    )

    def test_positive_selector_formula_neutrality_and_residual_bound(self) -> None:
        full = TPDCleanV5Block(
            2,
            activate=False,
            use_context_selector=True,
        )
        control = TPDCleanV5Block(
            2,
            activate=False,
            use_context_selector=False,
        )
        with torch.no_grad():
            full.phase_compress.weight.fill_(0.125)
            full.phase_compress.bias.zero_()
            full.saliency_scale.copy_(torch.tensor([0.4, -0.3]))
        incompatible = control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

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
            ]
        )
        outputs = []
        codes = []
        for block in (full, control):
            keep, context, saliency = block.branches(inputs)
            code = block.context_code(context)
            selector = block.context_selector(context)
            expected = saliency * torch.tanh(
                block.saliency_scale.view(1, -1, 1, 1) * selector
            )
            actual_keep, residual, actual_saliency, actual_selector = (
                block.fusion_terms(inputs)
            )
            with self.subTest(selector=block.use_context_selector):
                torch.testing.assert_close(actual_keep, keep)
                torch.testing.assert_close(actual_saliency, saliency)
                torch.testing.assert_close(actual_selector, selector)
                torch.testing.assert_close(residual, expected)
                torch.testing.assert_close(block(inputs), keep + expected)
                self.assertGreaterEqual(
                    float(selector.min()),
                    CONTEXT_SELECTOR_FLOOR - 1e-7,
                )
                self.assertLessEqual(
                    float(selector.max()),
                    CONTEXT_SELECTOR_CEILING + 1e-7,
                )
                self.assertTrue(
                    torch.all(residual.abs() <= saliency.abs() + 1e-7)
                )
                for channel, sign in ((0, 1), (1, -1)):
                    nonzero = saliency[:, channel] > 0
                    signed = residual[:, channel][nonzero] * sign
                    self.assertTrue(torch.all(signed > 0))
            outputs.append(block(inputs))
            codes.append(code)
        torch.testing.assert_close(codes[0], codes[1], rtol=0.0, atol=0.0)
        self.assertGreater(
            float((outputs[0] - outputs[1]).abs().max().detach()),
            1e-6,
        )

        constant = torch.full((1, 2, 4, 5), 3.25)
        self.assertTrue(
            torch.equal(
                full.context_code(constant),
                torch.zeros_like(constant),
            )
        )
        self.assertTrue(
            torch.equal(
                full.context_selector(constant),
                torch.ones_like(constant),
            )
        )
        self.assertTrue(
            torch.equal(
                control.context_code(constant),
                torch.zeros_like(constant),
            )
        )
        self.assertTrue(
            torch.equal(
                control.context_selector(constant),
                torch.ones_like(constant),
            )
        )

    def test_zero_scale_and_formal_initializer_are_exactly_spd(self) -> None:
        for variant in SUPPORTED_CLEAN_V5_VARIANTS:
            torch.manual_seed(17)
            spd = SPDPatchEmbedding(channels=4, stride=8)
            torch.manual_seed(17)
            candidate = build_clean_v5_patch_embedding(
                variant,
                channels=4,
                stride=8,
            )
            torch.manual_seed(29)
            spd.apply(weights_init_kaiming)
            torch.manual_seed(29)
            candidate.apply(weights_init_kaiming)
            spd_x = torch.randn(2, 4, 32, 40)
            candidate_x = spd_x.clone()
            for spd_block, candidate_block in zip(
                spd.blocks,
                candidate.blocks,
            ):
                with self.subTest(variant=variant):
                    self.assertEqual(
                        torch.count_nonzero(
                            candidate_block.saliency_scale
                        ).item(),
                        0,
                    )
                    self.assertTrue(
                        torch.equal(
                            candidate_block.phase_compress.weight,
                            spd_block.phase_compress.weight,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            candidate_block.phase_compress.bias,
                            spd_block.phase_compress.bias,
                        )
                    )
                spd_x = spd_block(spd_x)
                candidate_x = candidate_block(candidate_x)
                self.assertTrue(torch.equal(candidate_x, spd_x))

    def test_zero_scale_gradient_matches_positive_selector_derivative(
        self,
    ) -> None:
        spatial_weight = torch.tensor(
            [[[[1.0, -0.5], [0.25, 1.5]]]],
            dtype=torch.float32,
        )
        inputs_template = torch.tensor(
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
            ]
        )
        for variant in SUPPORTED_CLEAN_V5_VARIANTS:
            spec = clean_v5_variant_spec(variant)
            block = TPDCleanV5Block(
                2,
                activate=False,
                use_context_selector=(
                    spec["context_selector"]
                    == "positive_centered_0p5_to_1p5"
                ),
            )
            with torch.no_grad():
                block.phase_compress.weight.fill_(0.125)
                block.phase_compress.bias.zero_()
            inputs = inputs_template.clone().requires_grad_(True)
            _, context, saliency = block.branches(inputs)
            selector = block.context_selector(context)
            expected = (
                spatial_weight * saliency * selector
            ).sum(dim=(0, 2, 3))
            loss = (block(inputs) * spatial_weight).sum()
            loss.backward()
            with self.subTest(variant=variant):
                torch.testing.assert_close(
                    block.saliency_scale.grad,
                    expected,
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertTrue(torch.isfinite(inputs.grad).all())
                for name, parameter in block.named_parameters():
                    self.assertIsNotNone(parameter.grad, msg=name)
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all(),
                        msg=name,
                    )
                    self.assertGreater(
                        float(parameter.grad.abs().sum()),
                        0.0,
                        msg=name,
                    )

    def test_context_selector_forms_a_stable_two_step_gradient_chain(
        self,
    ) -> None:
        block = TPDCleanV5Block(
            1,
            activate=False,
            use_context_selector=True,
        )
        context0 = torch.tensor(
            [[[[0.2, 1.5], [0.7, 3.0]]]],
            requires_grad=True,
        )
        selector0 = block.context_selector(context0)
        residual0 = torch.tanh(
            block.saliency_scale.view(1, 1, 1, 1) * selector0
        ).sum()
        scale_gradient = torch.autograd.grad(
            residual0,
            block.saliency_scale,
            retain_graph=True,
        )[0]
        context_gradient0 = torch.autograd.grad(
            residual0,
            context0,
        )[0]
        self.assertGreater(float(scale_gradient.abs().sum()), 0.0)
        self.assertEqual(float(context_gradient0.abs().sum()), 0.0)

        with torch.no_grad():
            block.saliency_scale.fill_(0.01)
        context1 = context0.detach().clone().requires_grad_(True)
        residual1 = torch.tanh(
            block.saliency_scale.view(1, 1, 1, 1)
            * block.context_selector(context1)
        ).sum()
        context_gradient1 = torch.autograd.grad(residual1, context1)[0]
        self.assertTrue(torch.isfinite(context_gradient1).all())
        self.assertGreater(float(context_gradient1.abs().sum()), 0.0)

    def test_cpu_autocast_forward_backward_is_finite(self) -> None:
        block = TPDCleanV5Block(
            4,
            activate=True,
            use_context_selector=True,
        )
        with torch.no_grad():
            block.saliency_scale.fill_(0.02)
        inputs = torch.randn(2, 4, 32, 48, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            keep, context, _ = block.branches(inputs)
            code = block.context_code(context)
            selector = block.context_selector(context)
            output = block(inputs)
        self.assertEqual(keep.dtype, torch.bfloat16)
        self.assertEqual(code.dtype, torch.float32)
        self.assertEqual(selector.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(block.saliency_scale.grad).all())
        self.assertGreater(float(block.saliency_scale.grad.abs().sum()), 0.0)

    def test_half_context_code_and_selector_remain_fp32(self) -> None:
        block = TPDCleanV5Block(
            2,
            activate=False,
            use_context_selector=True,
        )
        context = torch.tensor(
            [
                [
                    [[0.125, 0.53125], [1.1875, 3.25]],
                    [[-2.125, -0.375], [0.8125, 4.5]],
                ]
            ],
            dtype=torch.float16,
        )
        context_fp32 = context.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        expected_code = torch.tanh(
            centered
            * torch.rsqrt(
                centered.square().mean(
                    dim=(-2, -1),
                    keepdim=True,
                )
                + block.eps
            )
        )
        code = block.context_code(context)
        selector = block.context_selector(context)
        self.assertEqual(code.dtype, torch.float32)
        self.assertEqual(selector.dtype, torch.float32)
        torch.testing.assert_close(code, expected_code, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            selector,
            1.0 + 0.5 * expected_code,
            rtol=0.0,
            atol=0.0,
        )

    def test_paired_state_layout_parameter_count_and_strict_reload(self) -> None:
        layouts = []
        for variant in SUPPORTED_CLEAN_V5_VARIANTS:
            emb1 = build_clean_v5_patch_embedding(variant, 32, 16)
            emb2 = build_clean_v5_patch_embedding(variant, 64, 8)
            self.assertEqual(
                parameter_count(emb1) + parameter_count(emb2),
                66_176,
            )
            layouts.append(
                tuple(
                    (name, tuple(value.shape), value.dtype)
                    for prefix, embedding in (
                        ("embeddings_1", emb1),
                        ("embeddings_2", emb2),
                    )
                    for name, value in embedding.state_dict().items()
                )
            )

            model = build_clean_v5_patch_embedding(variant, 4, 8)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.01)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            rebuilt = build_clean_v5_patch_embedding(variant, 4, 8)
            incompatible = rebuilt.load_state_dict(
                torch.load(buffer, map_location="cpu", weights_only=True),
                strict=True,
            )
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            inputs = torch.randn(2, 4, 32, 40)
            self.assertTrue(torch.equal(rebuilt(inputs), model(inputs)))
        self.assertEqual(layouts[0], layouts[1])

        full = build_clean_v5_patch_embedding(
            "tpd_clean_v5_full",
            4,
            8,
        )
        control = build_clean_v5_patch_embedding(
            "tpd_clean_v5_sal_capacity",
            4,
            8,
        )
        incompatible = control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_none_and_invalid_inputs_are_rejected(self) -> None:
        embedding = build_clean_v5_patch_embedding(
            PRIMARY_CLEAN_V5_VARIANT,
            4,
            4,
        )
        self.assertIsNone(embedding(None))
        with self.assertRaisesRegex(ValueError, "even H/W"):
            embedding(torch.randn(1, 4, 31, 32))
        with self.assertRaisesRegex(ValueError, "expected 4 channels"):
            embedding(torch.randn(1, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "BxCxHxW"):
            embedding(torch.randn(4, 32, 32))
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_clean_v5_patch_embedding(
                PRIMARY_CLEAN_V5_VARIANT,
                4,
                6,
            )
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v5"):
            clean_v5_variant_spec("unknown")


if __name__ == "__main__":
    unittest.main()
