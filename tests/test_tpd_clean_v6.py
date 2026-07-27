from __future__ import annotations

import io
import unittest

import torch
import torch.nn.functional as F

from experiments.train_tpd_pilot import weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd import SPDPatchEmbedding, replace_shallow_embeddings
from model.tpd_clean_v5 import build_clean_v5_patch_embedding
from model.tpd_clean_v6 import (
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    PRIMARY_CLEAN_V6_VARIANT,
    SUPPORTED_CLEAN_V6_VARIANTS,
    TPDCleanV6Block,
    TPDCleanV6PatchEmbedding,
    build_clean_v6_patch_embedding,
    clean_v6_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v6,
)


def _build_full_model(variant: str, seed: int) -> SCTransNet:
    """CPU-only paired builder matching the established initialization order."""

    torch.manual_seed(seed)
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v6(model, variant)
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


def _build_spd_model(seed: int) -> SCTransNet:
    """CPU-only dense-SPD reference with the same initialization order."""

    torch.manual_seed(seed)
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(weights_init_kaiming)
    replacements = replace_shallow_embeddings(model, "spd")
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


class TPDCleanV6Tests(unittest.TestCase):
    def test_variant_metadata_keeps_kcs_contract(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V6_VARIANTS,
            (
                "tpd_clean_v6_full",
                "tpd_clean_v6_phase_capacity",
            ),
        )
        self.assertEqual(PRIMARY_CLEAN_V6_VARIANT, "tpd_clean_v6_full")

        full = clean_v6_variant_spec("tpd_clean_v6_full")
        control = clean_v6_variant_spec("tpd_clean_v6_phase_capacity")
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        self.assertEqual(
            full["context_modulation"],
            "half_centered_context_code",
        )
        self.assertEqual(control["context_modulation"], "zero")
        self.assertEqual(full["context_headroom"], (
            "one_plus_half_one_minus_abs_scale_times_modulation"
        ))
        self.assertEqual(control["context_headroom"], "neutral_one")

        for spec in (full, control):
            self.assertEqual(
                spec["candidate_family"],
                "spd_anchored_tpd_clean_v6_phase_tied_kcs_zero_mean_gain",
            )
            self.assertEqual(spec["mainline_contract"], "Keep-Context-Saliency")
            self.assertEqual(
                spec["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertFalse(spec["fourth_parallel_branch_added"])
            self.assertEqual(spec["learned_scales_per_block"], 1)
            self.assertEqual(
                spec["scale_parameter"],
                "per_channel_saliency_scale",
            )
            self.assertEqual(
                spec["phase_tied_projection"],
                "sum_keep_weights_over_four_contiguous_phases",
            )
            self.assertEqual(
                spec["pixel_unshuffle_channel_order"],
                "input_channel_major_four_phases_contiguous",
            )
            self.assertEqual(spec["zero_scale_reference"], "dense_spd_exact")
            self.assertEqual(spec["shallow_embedding_parameters"], 66_176)
            self.assertEqual(spec["full_model_parameters"], 10_843_155)

    def test_pixel_unshuffle_order_and_phase_tied_weight_are_exact(self) -> None:
        inputs = torch.zeros(1, 2, 2, 2)
        inputs[0, 0] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        inputs[0, 1] = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
        unshuffled = F.pixel_unshuffle(inputs, 2)
        self.assertEqual(
            unshuffled.flatten().tolist(),
            [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        )

        block = TPDCleanV6Block(
            2,
            activate=False,
            use_context_headroom=True,
        )
        with torch.no_grad():
            block.phase_compress.weight.copy_(
                torch.arange(16, dtype=torch.float32).reshape(2, 8, 1, 1)
            )
            block.phase_compress.bias.zero_()
        expected = (
            block.phase_compress.weight
            .reshape(2, 2, 4, 1, 1)
            .sum(dim=2)
        )
        actual = block.phase_tied_weight()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        # When all four phases are equal, dense Keep must equal the derived
        # projection exactly.  This independently verifies the reshape axis.
        low_resolution = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]
        )
        phase_constant = low_resolution.repeat_interleave(
            2,
            dim=-2,
        ).repeat_interleave(2, dim=-1)
        dense_keep = block.phase_compress(
            F.pixel_unshuffle(phase_constant, 2)
        )
        tied = F.conv2d(low_resolution, actual, bias=None)
        torch.testing.assert_close(dense_keep, tied, rtol=0.0, atol=0.0)

    def test_three_sources_alignment_formula_and_control_difference(self) -> None:
        full = TPDCleanV6Block(
            2,
            activate=False,
            use_context_headroom=True,
        )
        control = TPDCleanV6Block(
            2,
            activate=False,
            use_context_headroom=False,
        )
        with torch.no_grad():
            full.phase_compress.weight.copy_(
                torch.linspace(-0.4, 0.6, 16).reshape(2, 8, 1, 1)
            )
            full.phase_compress.bias.copy_(torch.tensor([0.1, -0.2]))
            full.saliency_scale.copy_(torch.tensor([0.45, -0.35]))
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
        aligned_sources = []
        for block in (full, control):
            keep, context0, saliency0 = block.branches(inputs)
            tied_weight = block.phase_tied_weight()
            expected_context = F.conv2d(
                context0.float(),
                tied_weight,
                bias=None,
            )
            expected_saliency = F.conv2d(
                saliency0.float(),
                tied_weight,
                bias=None,
            )
            actual_keep, context_aligned, saliency_aligned = (
                block.aligned_branches(inputs)
            )
            torch.testing.assert_close(actual_keep, keep, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                context_aligned,
                expected_context,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                saliency_aligned,
                expected_saliency,
                rtol=0.0,
                atol=0.0,
            )

            code = block.context_code(context_aligned)
            expected_modulation = (
                0.5
                * (code - code.mean(dim=(-2, -1), keepdim=True))
                if block.use_context_headroom
                else torch.zeros_like(code)
            )
            scale = torch.tanh(block.saliency_scale).view(1, -1, 1, 1)
            expected_headroom = (
                1.0
                + 0.5
                * (1.0 - scale.abs())
                * expected_modulation
            )
            expected_residual = saliency_aligned * (
                scale * expected_headroom
            )
            (
                fused_keep,
                residual,
                fused_saliency,
                modulation,
            ) = block.fusion_terms(inputs)
            actual_scale, actual_modulation, actual_headroom = block.headroom(
                context_aligned
            )
            torch.testing.assert_close(fused_keep, keep, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                fused_saliency,
                saliency_aligned,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                actual_scale,
                scale,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                modulation,
                expected_modulation,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                actual_modulation,
                expected_modulation,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                actual_headroom,
                expected_headroom,
                rtol=1e-6,
                atol=1e-7,
            )
            torch.testing.assert_close(
                residual,
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
            self.assertLessEqual(
                float(actual_headroom.max().detach()),
                CONTEXT_HEADROOM_CEILING + 1e-7,
            )
            self.assertGreaterEqual(
                float(actual_headroom.min().detach()),
                CONTEXT_HEADROOM_FLOOR - 1e-7,
            )
            self.assertTrue(
                torch.all(
                    (actual_scale * actual_headroom).abs() <= 1.0 + 1e-7
                )
            )
            self.assertTrue(
                torch.all(residual.abs() <= saliency_aligned.abs() + 1e-7)
            )
            outputs.append(block(inputs))
            aligned_sources.append((keep, context_aligned, saliency_aligned))

        for full_source, control_source in zip(
            aligned_sources[0],
            aligned_sources[1],
        ):
            torch.testing.assert_close(
                full_source,
                control_source,
                rtol=0.0,
                atol=0.0,
            )
        self.assertGreater(
            float((outputs[0] - outputs[1]).abs().max().detach()),
            1e-6,
        )
        full_modulation = full.context_modulation(aligned_sources[0][1])
        control_modulation = control.context_modulation(aligned_sources[1][1])
        torch.testing.assert_close(
            full_modulation.mean(dim=(-2, -1)),
            torch.zeros_like(full_modulation.mean(dim=(-2, -1))),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertGreater(float(full_modulation.abs().sum().detach()), 0.0)
        self.assertEqual(float(control_modulation.abs().sum().detach()), 0.0)

    def test_zero_saliency_support_has_zero_residual(self) -> None:
        block = TPDCleanV6Block(
            2,
            activate=False,
            use_context_headroom=True,
        )
        with torch.no_grad():
            block.saliency_scale.copy_(torch.tensor([0.7, -0.6]))
        low_resolution = torch.randn(1, 2, 3, 4)
        inputs = low_resolution.repeat_interleave(
            2,
            dim=-2,
        ).repeat_interleave(2, dim=-1)
        _, _, saliency0 = block.branches(inputs)
        _, residual, saliency_aligned, _ = block.fusion_terms(inputs)
        self.assertTrue(torch.equal(saliency0, torch.zeros_like(saliency0)))
        self.assertTrue(
            torch.equal(saliency_aligned, torch.zeros_like(saliency_aligned))
        )
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_zero_gate_is_bit_exact_dense_spd(self) -> None:
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            torch.manual_seed(17)
            spd = SPDPatchEmbedding(channels=4, stride=8)
            torch.manual_seed(17)
            candidate = build_clean_v6_patch_embedding(
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
                        tuple(candidate_block.saliency_scale.shape),
                        (4,),
                    )
                    self.assertEqual(
                        int(torch.count_nonzero(
                            candidate_block.saliency_scale
                        )),
                        0,
                    )
                    self.assertTrue(torch.equal(
                        candidate_block.phase_compress.weight,
                        spd_block.phase_compress.weight,
                    ))
                    self.assertTrue(torch.equal(
                        candidate_block.phase_compress.bias,
                        spd_block.phase_compress.bias,
                    ))
                spd_x = spd_block(spd_x)
                candidate_x = candidate_block(candidate_x)
                self.assertTrue(torch.equal(candidate_x, spd_x))

    def test_shapes_none_and_invalid_inputs(self) -> None:
        cases = (
            (32, 16, 32, 48, 4),
            (64, 8, 24, 32, 3),
        )
        for channels, stride, height, width, expected_blocks in cases:
            inputs = torch.randn(1, channels, height, width)
            for variant in SUPPORTED_CLEAN_V6_VARIANTS:
                embedding = build_clean_v6_patch_embedding(
                    variant,
                    channels,
                    stride,
                )
                with self.subTest(variant=variant, channels=channels):
                    self.assertIsInstance(embedding, TPDCleanV6PatchEmbedding)
                    self.assertEqual(len(embedding.blocks), expected_blocks)
                    self.assertEqual(
                        tuple(embedding(inputs).shape),
                        (
                            1,
                            channels,
                            height // stride,
                            width // stride,
                        ),
                    )
                    self.assertIsNone(embedding(None))

        embedding = build_clean_v6_patch_embedding(
            PRIMARY_CLEAN_V6_VARIANT,
            4,
            4,
        )
        with self.assertRaisesRegex(ValueError, "even H/W"):
            embedding(torch.randn(1, 4, 31, 32))
        with self.assertRaisesRegex(ValueError, "expected 4 channels"):
            embedding(torch.randn(1, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "BxCxHxW"):
            embedding(torch.randn(4, 32, 32))
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_clean_v6_patch_embedding(
                PRIMARY_CLEAN_V6_VARIANT,
                4,
                6,
            )
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v6"):
            clean_v6_variant_spec("unknown")

    def test_state_keys_parameter_counts_and_v5_compatibility(self) -> None:
        layouts = []
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            emb1 = build_clean_v6_patch_embedding(variant, 32, 16)
            emb2 = build_clean_v6_patch_embedding(variant, 64, 8)
            self.assertEqual(
                parameter_count(emb1) + parameter_count(emb2),
                66_176,
            )
            layouts.append(tuple(
                (prefix + "." + name, tuple(tensor.shape), tensor.dtype)
                for prefix, embedding in (
                    ("embeddings_1", emb1),
                    ("embeddings_2", emb2),
                )
                for name, tensor in embedding.state_dict().items()
            ))
            for embedding in (emb1, emb2):
                for block in embedding.blocks:
                    self.assertEqual(
                        set(dict(block.named_parameters())),
                        {
                            "saliency_scale",
                            "phase_compress.weight",
                            "phase_compress.bias",
                        },
                    )
                    self.assertEqual(
                        set(dict(block.named_children())),
                        {"phase_compress", "activation"},
                    )
                    self.assertEqual(dict(block.named_buffers()), {})
                    self.assertEqual(
                        tuple(block.saliency_scale.shape),
                        (block.channels,),
                    )
                    self.assertFalse(hasattr(block, "context_projection"))
                    self.assertFalse(hasattr(block, "saliency_projection"))
        self.assertEqual(layouts[0], layouts[1])

        v5 = build_clean_v5_patch_embedding(
            "tpd_clean_v5_full",
            channels=4,
            stride=8,
        )
        v6 = build_clean_v6_patch_embedding(
            PRIMARY_CLEAN_V6_VARIANT,
            channels=4,
            stride=8,
        )
        self.assertEqual(tuple(v6.state_dict()), tuple(v5.state_dict()))
        for key in v5.state_dict():
            self.assertEqual(
                v6.state_dict()[key].shape,
                v5.state_dict()[key].shape,
                msg=key,
            )
        incompatible = v6.load_state_dict(v5.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        full = build_clean_v6_patch_embedding(
            "tpd_clean_v6_full",
            channels=4,
            stride=8,
        )
        control = build_clean_v6_patch_embedding(
            "tpd_clean_v6_phase_capacity",
            channels=4,
            stride=8,
        )
        incompatible = control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        full_model = _build_full_model(PRIMARY_CLEAN_V6_VARIANT, seed=42)
        self.assertEqual(parameter_count(full_model), 10_843_155)
        self.assertIsInstance(
            full_model.mtc.embeddings_1,
            TPDCleanV6PatchEmbedding,
        )
        self.assertIsInstance(
            full_model.mtc.embeddings_2,
            TPDCleanV6PatchEmbedding,
        )

    def test_full_and_control_have_identical_paired_initial_state(self) -> None:
        full = _build_full_model("tpd_clean_v6_full", seed=3407)
        control = _build_full_model(
            "tpd_clean_v6_phase_capacity",
            seed=3407,
        )
        full_state = full.state_dict()
        control_state = control.state_dict()
        self.assertEqual(tuple(full_state), tuple(control_state))
        for key in full_state:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(full_state[key], control_state[key]))

        module_names = tuple(name.lower() for name, _ in full.named_modules())
        self.assertFalse(any("relay" in name or "ner" in name for name in module_names))
        self.assertTrue(all(not module._forward_hooks for module in full.modules()))
        self.assertTrue(
            all(not module._forward_pre_hooks for module in full.modules())
        )

    def test_full_network_zero_gate_is_bit_exact_spd(self) -> None:
        spd = _build_spd_model(seed=101)
        inputs = torch.randn(1, 1, 32, 32)
        spd.eval()
        with torch.inference_mode():
            expected = spd(inputs)
        self.assertEqual(len(expected), 6)

        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            candidate = _build_full_model(variant, seed=101)
            candidate.eval()
            with torch.inference_mode():
                actual = candidate(inputs)
            with self.subTest(variant=variant):
                self.assertEqual(len(actual), 6)
                for output, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(output, reference))

    def test_cpu_forward_backward_gradients_and_strict_reload(self) -> None:
        for variant in SUPPORTED_CLEAN_V6_VARIANTS:
            embedding = build_clean_v6_patch_embedding(
                variant,
                channels=4,
                stride=4,
            )
            with torch.no_grad():
                for index, block in enumerate(embedding.blocks):
                    block.saliency_scale.copy_(
                        torch.tensor([0.08, -0.06, 0.05, -0.04])
                        * (index + 1)
                    )
            inputs = torch.randn(
                2,
                4,
                16,
                20,
                requires_grad=True,
            )
            output = embedding(inputs)
            self.assertTrue(torch.isfinite(output).all())
            loss = output.square().mean()
            loss.backward()
            self.assertIsNotNone(inputs.grad)
            self.assertTrue(torch.isfinite(inputs.grad).all())
            self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
            for name, parameter in embedding.named_parameters():
                with self.subTest(variant=variant, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

            state_buffer = io.BytesIO()
            torch.save(embedding.state_dict(), state_buffer)
            state_buffer.seek(0)
            rebuilt = build_clean_v6_patch_embedding(
                variant,
                channels=4,
                stride=4,
            )
            incompatible = rebuilt.load_state_dict(
                torch.load(
                    state_buffer,
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            check_inputs = torch.randn(2, 4, 16, 20)
            self.assertTrue(
                torch.equal(rebuilt(check_inputs), embedding(check_inputs))
            )


if __name__ == "__main__":
    unittest.main()
