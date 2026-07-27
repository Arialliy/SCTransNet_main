from __future__ import annotations

import io
import unittest

import torch
import torch.nn.functional as F

from experiments.train_tpd_pilot import weights_init_kaiming
from model.tpd import SPDPatchEmbedding
from model.tpd_clean_v6 import (
    TPDCleanV6Block,
    build_clean_v6_patch_embedding,
)
from model.tpd_clean_v7 import (
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    PRIMARY_CLEAN_V7_VARIANT,
    SUPPORTED_CLEAN_V7_VARIANTS,
    TPDCleanV7Block,
    TPDCleanV7PatchEmbedding,
    build_clean_v7_patch_embedding,
    clean_v7_variant_spec,
    parameter_count,
)


class TPDCleanV7Tests(unittest.TestCase):
    def test_variant_contract_preserves_three_source_mainline(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V7_VARIANTS,
            ("tpd_clean_v7_full", "tpd_clean_v7_phase_capacity"),
        )
        self.assertEqual(PRIMARY_CLEAN_V7_VARIANT, "tpd_clean_v7_full")
        full = clean_v7_variant_spec("tpd_clean_v7_full")
        control = clean_v7_variant_spec("tpd_clean_v7_phase_capacity")
        self.assertEqual(full["context_gate"], 1.0)
        self.assertEqual(control["context_gate"], 0.0)
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(control["primary_candidate"])
        for spec in (full, control):
            self.assertEqual(spec["mainline_contract"], "Keep-Context-Saliency")
            self.assertEqual(
                spec["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertFalse(spec["fourth_parallel_branch_added"])
            self.assertEqual(spec["saliency_formula"], "D_p=relu(Z_p-C0)")
            self.assertEqual(
                spec["saliency_projection"],
                "complete_keep_weight_phase_projection",
            )
            self.assertEqual(spec["state_compatible_with"], "tpd_clean_v6")
            self.assertEqual(spec["shallow_embedding_parameters"], 66_176)
            self.assertEqual(spec["full_model_parameters"], 10_843_155)

    def test_phase_order_saliency_and_projection_match_hand_calculation(
        self,
    ) -> None:
        block = TPDCleanV7Block(
            1,
            activate=False,
            context_gate=1.0,
        )
        inputs = torch.tensor([[[[1.0, 4.0], [2.0, 3.0]]]])
        with torch.no_grad():
            block.phase_compress.weight.copy_(
                torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(1, 4, 1, 1)
            )
            block.phase_compress.bias.fill_(0.25)

        rearranged, context, saliency_phases = block.phase_sources(inputs)
        self.assertEqual(rearranged.flatten().tolist(), [1.0, 4.0, 2.0, 3.0])
        torch.testing.assert_close(
            context,
            torch.tensor([[[[2.5]]]]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            saliency_phases.flatten(),
            torch.tensor([0.0, 1.5, 0.0, 0.5]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            saliency_phases.max(dim=2).values,
            F.max_pool2d(inputs, 2, 2) - F.avg_pool2d(inputs, 2, 2),
            rtol=0.0,
            atol=0.0,
        )

        keep, context_aligned, saliency_aligned = block.aligned_branches(inputs)
        torch.testing.assert_close(
            keep,
            torch.tensor([[[[41.25]]]]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            context_aligned,
            torch.tensor([[[[37.5]]]]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            saliency_aligned,
            torch.tensor([[[[7.0]]]]),
            rtol=0.0,
            atol=0.0,
        )

    def test_phase_resolved_projection_distinguishes_peak_location(self) -> None:
        v6 = TPDCleanV6Block(
            1,
            activate=False,
            use_context_headroom=True,
        )
        v7 = TPDCleanV7Block(
            1,
            activate=False,
            context_gate=1.0,
        )
        with torch.no_grad():
            weights = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(
                1, 4, 1, 1
            )
            v6.phase_compress.weight.copy_(weights)
            v6.phase_compress.bias.zero_()
        v7.load_state_dict(v6.state_dict(), strict=True)

        peak_tl = torch.tensor([[[[4.0, 0.0], [0.0, 0.0]]]])
        peak_tr = torch.tensor([[[[0.0, 4.0], [0.0, 0.0]]]])
        v6_sa = [
            v6.aligned_branches(inputs)[2]
            for inputs in (peak_tl, peak_tr)
        ]
        v7_sa = [
            v7.aligned_branches(inputs)[2]
            for inputs in (peak_tl, peak_tr)
        ]
        torch.testing.assert_close(v6_sa[0], v6_sa[1], rtol=0.0, atol=0.0)
        self.assertEqual(float(v7_sa[0]), 3.0)
        self.assertEqual(float(v7_sa[1]), 6.0)

    def test_v6_and_v7_keep_context_and_headroom_are_identical(self) -> None:
        v6 = TPDCleanV6Block(
            3,
            activate=False,
            use_context_headroom=True,
        )
        v7 = TPDCleanV7Block(
            3,
            activate=False,
            context_gate=1.0,
        )
        torch.manual_seed(71)
        v6.apply(weights_init_kaiming)
        with torch.no_grad():
            v6.saliency_scale.copy_(torch.tensor([0.3, -0.2, 0.1]))
        v7.load_state_dict(v6.state_dict(), strict=True)
        inputs = torch.randn(2, 3, 8, 10)

        v6_keep, v6_context, v6_saliency = v6.aligned_branches(inputs)
        v7_keep, v7_context, v7_saliency = v7.aligned_branches(inputs)
        self.assertTrue(torch.equal(v6_keep, v7_keep))
        self.assertTrue(torch.equal(v6_context, v7_context))
        self.assertGreater(
            float((v6_saliency - v7_saliency).abs().sum()),
            0.0,
        )
        for v6_term, v7_term in zip(
            v6.headroom(v6_context),
            v7.headroom(v7_context),
        ):
            self.assertTrue(torch.equal(v6_term, v7_term))

    def test_full_and_capacity_share_sources_and_differ_only_in_headroom(
        self,
    ) -> None:
        full = TPDCleanV7Block(2, activate=False, context_gate=1.0)
        control = TPDCleanV7Block(2, activate=False, context_gate=0.0)
        torch.manual_seed(83)
        full.apply(weights_init_kaiming)
        with torch.no_grad():
            full.saliency_scale.copy_(torch.tensor([0.45, -0.35]))
        incompatible = control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        inputs = torch.randn(2, 2, 8, 10)

        full_sources = full.aligned_branches(inputs)
        control_sources = control.aligned_branches(inputs)
        for lhs, rhs in zip(full_sources, control_sources):
            self.assertTrue(torch.equal(lhs, rhs))
        full_scale, full_modulation, full_headroom = full.headroom(
            full_sources[1]
        )
        control_scale, control_modulation, control_headroom = control.headroom(
            control_sources[1]
        )
        self.assertTrue(torch.equal(full_scale, control_scale))
        self.assertGreater(float(full_modulation.abs().sum()), 0.0)
        self.assertEqual(float(control_modulation.abs().sum()), 0.0)
        self.assertTrue(torch.equal(
            control_headroom,
            torch.ones_like(control_headroom),
        ))
        self.assertGreater(
            float((full_headroom - control_headroom).abs().sum()),
            0.0,
        )
        self.assertGreater(
            float((full(inputs) - control(inputs)).abs().sum()),
            0.0,
        )

    def test_state_layout_counts_and_v6_strict_compatibility(self) -> None:
        layouts = []
        for variant in SUPPORTED_CLEAN_V7_VARIANTS:
            emb1 = build_clean_v7_patch_embedding(variant, 32, 16)
            emb2 = build_clean_v7_patch_embedding(variant, 64, 8)
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
        self.assertEqual(layouts[0], layouts[1])

        v6 = build_clean_v6_patch_embedding(
            "tpd_clean_v6_full",
            channels=4,
            stride=8,
        )
        v7 = build_clean_v7_patch_embedding(
            PRIMARY_CLEAN_V7_VARIANT,
            channels=4,
            stride=8,
        )
        self.assertEqual(tuple(v6.state_dict()), tuple(v7.state_dict()))
        incompatible = v7.load_state_dict(v6.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        incompatible = v6.load_state_dict(v7.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_zero_scale_is_bit_exact_spd_and_v6(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_VARIANTS:
            torch.manual_seed(17)
            spd = SPDPatchEmbedding(channels=4, stride=8)
            torch.manual_seed(17)
            v6 = build_clean_v6_patch_embedding(
                "tpd_clean_v6_full",
                channels=4,
                stride=8,
            )
            torch.manual_seed(17)
            v7 = build_clean_v7_patch_embedding(
                variant,
                channels=4,
                stride=8,
            )
            torch.manual_seed(29)
            spd.apply(weights_init_kaiming)
            torch.manual_seed(29)
            v6.apply(weights_init_kaiming)
            torch.manual_seed(29)
            v7.apply(weights_init_kaiming)
            inputs = torch.randn(2, 4, 32, 40)
            expected = spd(inputs)
            self.assertTrue(torch.equal(v6(inputs), expected))
            self.assertTrue(torch.equal(v7(inputs), expected))

    def test_zero_saliency_bounds_shapes_and_validation(self) -> None:
        block = TPDCleanV7Block(2, activate=False, context_gate=1.0)
        with torch.no_grad():
            block.saliency_scale.copy_(torch.tensor([0.7, -0.6]))
        low_resolution = torch.randn(1, 2, 3, 4)
        inputs = low_resolution.repeat_interleave(
            2, dim=-2
        ).repeat_interleave(2, dim=-1)
        _, _, phase_saliency = block.phase_sources(inputs)
        _, residual, aligned_saliency, _ = block.fusion_terms(inputs)
        self.assertTrue(torch.equal(
            phase_saliency,
            torch.zeros_like(phase_saliency),
        ))
        self.assertTrue(torch.equal(
            aligned_saliency,
            torch.zeros_like(aligned_saliency),
        ))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        random_inputs = torch.randn(2, 2, 8, 10)
        _, context, saliency = block.aligned_branches(random_inputs)
        scale, _, headroom = block.headroom(context)
        _, random_residual, _, _ = block.fusion_terms(random_inputs)
        self.assertLessEqual(
            float(headroom.max()),
            CONTEXT_HEADROOM_CEILING + 1e-7,
        )
        self.assertGreaterEqual(
            float(headroom.min()),
            CONTEXT_HEADROOM_FLOOR - 1e-7,
        )
        self.assertTrue(torch.all((scale * headroom).abs() <= 1.0 + 1e-7))
        self.assertTrue(
            torch.all(random_residual.abs() <= saliency.abs() + 1e-7)
        )

        embedding = build_clean_v7_patch_embedding(
            PRIMARY_CLEAN_V7_VARIANT,
            channels=4,
            stride=4,
        )
        self.assertIsInstance(embedding, TPDCleanV7PatchEmbedding)
        self.assertIsNone(embedding(None))
        self.assertEqual(
            tuple(embedding(torch.randn(1, 4, 16, 20)).shape),
            (1, 4, 4, 5),
        )
        with self.assertRaisesRegex(ValueError, "even H/W"):
            embedding(torch.randn(1, 4, 15, 20))
        with self.assertRaisesRegex(ValueError, "expected 4 channels"):
            embedding(torch.randn(1, 3, 16, 20))
        with self.assertRaisesRegex(ValueError, "BxCxHxW"):
            embedding(torch.randn(4, 16, 20))
        with self.assertRaisesRegex(ValueError, "power of two"):
            build_clean_v7_patch_embedding(
                PRIMARY_CLEAN_V7_VARIANT,
                channels=4,
                stride=6,
            )
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v7"):
            clean_v7_variant_spec("unknown")

    def test_cpu_gradients_and_strict_reload(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_VARIANTS:
            embedding = build_clean_v7_patch_embedding(
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
            inputs = torch.randn(2, 4, 16, 20, requires_grad=True)
            output = embedding(inputs)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
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
            rebuilt = build_clean_v7_patch_embedding(
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
