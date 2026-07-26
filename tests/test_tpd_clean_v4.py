from __future__ import annotations

import io
import unittest

import torch

from experiments.train_tpd_pilot import weights_init_kaiming
from model.tpd import SPDPatchEmbedding
from model.tpd_clean_v4 import (
    PRIMARY_CLEAN_V4_VARIANT,
    SUPPORTED_CLEAN_V4_VARIANTS,
    TPDCleanV4Block,
    TPDCleanV4PatchEmbedding,
    build_clean_v4_patch_embedding,
    clean_v4_variant_spec,
    parameter_count,
)


class TPDCleanV4Tests(unittest.TestCase):
    def test_frozen_candidate_matrix_and_primary(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V4_VARIANTS,
            (
                "tpd_clean_v4_full",
                "tpd_clean_v4_sal_capacity",
            ),
        )
        self.assertEqual(PRIMARY_CLEAN_V4_VARIANT, "tpd_clean_v4_full")
        self.assertEqual(
            {
                clean_v4_variant_spec(variant)["context_code"]
                for variant in SUPPORTED_CLEAN_V4_VARIANTS
            },
            {
                "centered_spatial_rms_tanh_fp32",
                "constant_one",
            },
        )
        for variant in SUPPORTED_CLEAN_V4_VARIANTS:
            spec = clean_v4_variant_spec(variant)
            with self.subTest(variant=variant):
                self.assertEqual(
                    spec["fusion_support"],
                    "single_bounded_saliency_logit",
                )
                self.assertEqual(
                    bool(spec["primary_candidate"]),
                    variant == PRIMARY_CLEAN_V4_VARIANT,
                )

    def test_dense_keep_three_sources_and_required_shapes(self) -> None:
        block = TPDCleanV4Block(
            channels=1,
            activate=False,
            use_context_code=True,
        )
        self.assertEqual(block.phase_compress.in_channels, 4)
        self.assertEqual(block.phase_compress.out_channels, 1)
        self.assertEqual(block.phase_compress.kernel_size, (1, 1))
        self.assertEqual(block.phase_compress.groups, 1)
        self.assertIsNotNone(block.phase_compress.bias)
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

        with torch.no_grad():
            block.phase_compress.weight.zero_()
            block.phase_compress.bias.zero_()
            block.phase_compress.weight[0, 0, 0, 0] = 1.0
        inputs = torch.zeros(1, 1, 4, 4)
        inputs[0, 0, 2, 2] = 4.0
        branches = block.branches(inputs)
        self.assertEqual(len(branches), 3)
        keep, context, saliency = branches
        expected_location = torch.tensor([[0, 0, 1, 1]])
        self.assertTrue(torch.equal(keep.nonzero(), expected_location))
        self.assertTrue(torch.equal(context.nonzero(), expected_location))
        self.assertTrue(torch.equal(saliency.nonzero(), expected_location))
        self.assertEqual(keep[0, 0, 1, 1].detach().item(), 4.0)
        self.assertEqual(float(context[0, 0, 1, 1]), 1.0)
        self.assertEqual(float(saliency[0, 0, 1, 1]), 3.0)

        cases = (
            (32, 16, 64, 96, 4),
            (64, 8, 48, 72, 3),
        )
        for channels, stride, height, width, expected_blocks in cases:
            candidate_inputs = torch.randn(1, channels, height, width)
            expected_shape = (
                1,
                channels,
                height // stride,
                width // stride,
            )
            for variant in SUPPORTED_CLEAN_V4_VARIANTS:
                embedding = build_clean_v4_patch_embedding(
                    variant,
                    channels,
                    stride,
                )
                with self.subTest(
                    variant=variant,
                    channels=channels,
                    stride=stride,
                ):
                    self.assertEqual(len(embedding.blocks), expected_blocks)
                    self.assertTrue(
                        all(
                            isinstance(item, TPDCleanV4Block)
                            for item in embedding.blocks
                        )
                    )
                    self.assertEqual(
                        tuple(embedding(candidate_inputs).shape),
                        expected_shape,
                    )

    def test_single_logit_formula_bound_and_control_semantics(self) -> None:
        full = TPDCleanV4Block(
            channels=2,
            activate=False,
            use_context_code=True,
        )
        control = TPDCleanV4Block(
            channels=2,
            activate=False,
            use_context_code=False,
        )
        with torch.no_grad():
            full.phase_compress.weight.fill_(0.125)
            full.phase_compress.bias.zero_()
            full.saliency_scale.copy_(torch.tensor([0.30, -0.20]))
            full.context_scale.copy_(torch.tensor([0.40, -0.50]))
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
        for block in (full, control):
            keep, context, saliency = block.branches(inputs)
            code = block.context_code(context)
            expected_logit = (
                block.saliency_scale.float().view(1, -1, 1, 1)
                + 0.5
                * torch.tanh(block.context_scale.float()).view(
                    1, -1, 1, 1
                )
                * code.float()
            )
            expected_residual = saliency * torch.tanh(expected_logit).to(
                saliency.dtype
            )
            (
                actual_keep,
                actual_residual,
                actual_saliency,
                actual_code,
            ) = block.fusion_terms(inputs)
            with self.subTest(use_context_code=block.use_context_code):
                torch.testing.assert_close(
                    actual_keep,
                    keep,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    actual_saliency,
                    saliency,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    actual_code,
                    code,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    actual_residual,
                    expected_residual,
                    rtol=1e-6,
                    atol=1e-7,
                )
                torch.testing.assert_close(
                    block(inputs),
                    keep + expected_residual,
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertTrue(
                    torch.all(
                        actual_residual.abs()
                        <= saliency.abs() + 1e-7
                    )
                )
                zero_support = saliency == 0
                self.assertTrue(
                    torch.equal(
                        actual_residual.masked_select(zero_support),
                        torch.zeros_like(
                            actual_residual.masked_select(zero_support)
                        ),
                    )
                )
            outputs.append(block(inputs))

        self.assertGreater(
            (outputs[0] - outputs[1]).abs().max().detach().item(),
            1e-6,
        )
        constant_context = torch.full((1, 2, 4, 5), 3.25)
        self.assertTrue(
            torch.equal(
                full.context_code(constant_context),
                torch.zeros_like(constant_context),
            )
        )
        self.assertTrue(
            torch.equal(
                control.context_code(constant_context),
                torch.ones_like(constant_context),
            )
        )

    def test_zero_scales_and_formal_initializer_are_exactly_spd(self) -> None:
        for variant in SUPPORTED_CLEAN_V4_VARIANTS:
            torch.manual_seed(17)
            spd = SPDPatchEmbedding(channels=4, stride=8)
            torch.manual_seed(17)
            candidate = build_clean_v4_patch_embedding(
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
                        int(
                            torch.count_nonzero(
                                candidate_block.context_scale
                            )
                        ),
                        0,
                    )
                    self.assertEqual(
                        int(
                            torch.count_nonzero(
                                candidate_block.saliency_scale
                            )
                        ),
                        0,
                    )
                    torch.testing.assert_close(
                        candidate_block.phase_compress.weight,
                        spd_block.phase_compress.weight,
                        rtol=0.0,
                        atol=0.0,
                    )
                    torch.testing.assert_close(
                        candidate_block.phase_compress.bias,
                        spd_block.phase_compress.bias,
                        rtol=0.0,
                        atol=0.0,
                    )
                spd_x = spd_block(spd_x)
                candidate_x = candidate_block(candidate_x)
                torch.testing.assert_close(
                    candidate_x,
                    spd_x,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_zero_scale_gradients_match_analytic_derivatives(self) -> None:
        spatial_weight = torch.tensor(
            [[[[1.0, -0.5], [0.25, 1.5]]]],
            dtype=torch.float32,
        )
        base_inputs = torch.tensor(
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
        for variant in SUPPORTED_CLEAN_V4_VARIANTS:
            spec = clean_v4_variant_spec(variant)
            block = TPDCleanV4Block(
                channels=2,
                activate=False,
                use_context_code=(
                    spec["context_code"]
                    == "centered_spatial_rms_tanh_fp32"
                ),
            )
            with torch.no_grad():
                block.phase_compress.weight.fill_(0.125)
                block.phase_compress.bias.zero_()
            inputs = base_inputs.clone().requires_grad_(True)
            _, context, saliency = block.branches(inputs)
            code = block.context_code(context)
            expected_saliency_gradient = (
                spatial_weight * saliency
            ).sum(dim=(0, 2, 3))
            expected_context_gradient = (
                0.5 * spatial_weight * saliency * code
            ).sum(dim=(0, 2, 3))

            loss = (block(inputs) * spatial_weight).sum()
            loss.backward()
            with self.subTest(variant=variant):
                torch.testing.assert_close(
                    block.saliency_scale.grad,
                    expected_saliency_gradient,
                    rtol=1e-6,
                    atol=1e-7,
                )
                torch.testing.assert_close(
                    block.context_scale.grad,
                    expected_context_gradient,
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
                for name, parameter in block.named_parameters():
                    with self.subTest(variant=variant, parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(
                            torch.isfinite(parameter.grad).all()
                        )
                        self.assertGreater(
                            float(parameter.grad.abs().sum()),
                            0.0,
                        )

    def test_capacity_state_layout_and_strict_round_trip(self) -> None:
        layouts = []
        for variant in SUPPORTED_CLEAN_V4_VARIANTS:
            emb1 = build_clean_v4_patch_embedding(variant, 32, 16)
            emb2 = build_clean_v4_patch_embedding(variant, 64, 8)
            self.assertEqual(
                parameter_count(emb1) + parameter_count(emb2),
                66_496,
            )
            layouts.append(
                tuple(
                    (name, tuple(tensor.shape), tensor.dtype)
                    for prefix, embedding in (
                        ("embeddings_1", emb1),
                        ("embeddings_2", emb2),
                    )
                    for name, tensor in embedding.state_dict().items()
                )
            )

            model = build_clean_v4_patch_embedding(
                variant,
                channels=4,
                stride=8,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * 0.01)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            rebuilt = build_clean_v4_patch_embedding(
                variant,
                channels=4,
                stride=8,
            )
            incompatible = rebuilt.load_state_dict(
                torch.load(
                    buffer,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            inputs = torch.randn(2, 4, 32, 40)
            torch.testing.assert_close(
                rebuilt(inputs),
                model(inputs),
                rtol=0.0,
                atol=0.0,
            )

        self.assertEqual(layouts[0], layouts[1])
        full = build_clean_v4_patch_embedding(
            "tpd_clean_v4_full",
            channels=4,
            stride=8,
        )
        control = build_clean_v4_patch_embedding(
            "tpd_clean_v4_sal_capacity",
            channels=4,
            stride=8,
        )
        incompatible = control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_none_and_invalid_input_contracts(self) -> None:
        embedding = build_clean_v4_patch_embedding(
            PRIMARY_CLEAN_V4_VARIANT,
            channels=4,
            stride=4,
        )
        self.assertIsNone(embedding(None))
        with self.assertRaisesRegex(ValueError, "even H/W"):
            embedding(torch.randn(1, 4, 31, 32))
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_clean_v4_patch_embedding(
                PRIMARY_CLEAN_V4_VARIANT,
                channels=4,
                stride=6,
            )
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v4"):
            clean_v4_variant_spec("unknown")


if __name__ == "__main__":
    unittest.main()
