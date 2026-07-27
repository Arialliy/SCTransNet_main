from __future__ import annotations

import io
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from experiments.train_tpd_pilot import weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd import SPDPatchEmbedding, replace_shallow_embeddings
from model.tpd_clean_v6 import (
    TPDCleanV6Block,
    build_clean_v6_patch_embedding,
)
from model.tpd_clean_v7_dch import (
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    PRIMARY_CLEAN_V7_DCH_VARIANT,
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
    TPDCleanV7DCHBlock,
    TPDCleanV7DCHPatchEmbedding,
    build_clean_v7_dch_patch_embedding,
    clean_v7_dch_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v7_dch,
)


torch.set_num_threads(1)


def _build_full_dch_model(variant: str, seed: int) -> SCTransNet:
    """Build a paired full model with the established initialization order."""

    torch.manual_seed(seed)
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v7_dch(model, variant)
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


def _build_full_spd_model(seed: int) -> SCTransNet:
    """Build the dense-SPD zero-scale reference with paired initialization."""

    torch.manual_seed(seed)
    model = SCTransNet(get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(weights_init_kaiming)
    replacements = replace_shallow_embeddings(model, "spd")
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


class TPDCleanV7DCHTests(unittest.TestCase):
    def assert_nested_exact(self, actual, expected, path: str = "root") -> None:
        if torch.is_tensor(actual):
            self.assertTrue(
                torch.equal(actual, expected),
                msg=f"tensor mismatch at {path}",
            )
            return
        self.assertIs(
            type(actual),
            type(expected),
            msg=f"type mismatch at {path}",
        )
        if isinstance(actual, dict):
            self.assertEqual(
                tuple(actual),
                tuple(expected),
                msg=f"key mismatch at {path}",
            )
            for key in actual:
                self.assert_nested_exact(
                    actual[key],
                    expected[key],
                    f"{path}.{key}",
                )
            return
        if isinstance(actual, (list, tuple)):
            self.assertEqual(
                len(actual),
                len(expected),
                msg=f"length mismatch at {path}",
            )
            for index, (item_a, item_b) in enumerate(zip(actual, expected)):
                self.assert_nested_exact(
                    item_a,
                    item_b,
                    f"{path}[{index}]",
                )
            return
        self.assertEqual(actual, expected, msg=f"value mismatch at {path}")

    def test_metadata_freezes_kcs_dch_and_auditable_fields(self) -> None:
        self.assertEqual(
            SUPPORTED_CLEAN_V7_DCH_VARIANTS,
            (
                "tpd_clean_v7_dch_full",
                "tpd_clean_v7_dch_capacity",
            ),
        )
        self.assertEqual(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            "tpd_clean_v7_dch_full",
        )

        full = clean_v7_dch_variant_spec(
            "TPD_CLEAN_V7_DCH_FULL",
        )
        capacity = clean_v7_dch_variant_spec(
            "tpd_clean_v7_dch_capacity",
        )
        self.assertEqual(full["context_gate"], 1.0)
        self.assertEqual(capacity["context_gate"], 0.0)
        self.assertTrue(full["primary_candidate"])
        self.assertFalse(capacity["primary_candidate"])
        self.assertEqual(
            full["context_code"],
            (
                "phase_aligned_centered_spatial_rms_eps_tanh_"
                "formal_amp_off_fp32"
            ),
        )
        self.assertEqual(
            capacity["context_code"],
            "not_computed_in_capacity_forward",
        )
        self.assertEqual(
            full["fusion_support"],
            (
                "phase_tied_deferred_zero_mean_context_gain_"
                "modulated_saliency"
            ),
        )
        self.assertEqual(
            capacity["fusion_support"],
            "phase_tied_saliency_capacity_control",
        )

        for spec in (full, capacity):
            self.assertEqual(
                spec["mainline_contract"],
                "Keep-Context-Saliency",
            )
            self.assertEqual(
                spec["semantic_sources"],
                ("Keep", "Context", "Saliency"),
            )
            self.assertFalse(spec["fourth_parallel_branch_added"])
            self.assertEqual(
                spec["phase_tied_projection"],
                "sum_keep_weights_over_four_contiguous_phases",
            )
            self.assertEqual(
                spec["pixel_unshuffle_channel_order"],
                "input_channel_major_four_phases_contiguous",
            )
            self.assertEqual(
                spec["saliency_representation"],
                "max_pool_minus_avg_pool_unchanged_from_v6",
            )
            self.assertEqual(spec["learned_scales_per_block"], 1)
            self.assertEqual(
                spec["scale_parameter"],
                "per_channel_saliency_scale",
            )
            self.assertEqual(spec["zero_scale_reference"], "dense_spd_exact")
            self.assertEqual(
                spec["zero_scale_first_order_reference"],
                "capacity_exact",
            )
            self.assertEqual(spec["state_compatible_with"], "tpd_clean_v6")
            self.assertEqual(spec["shallow_embedding_parameters"], 66_176)
            self.assertEqual(spec["full_model_parameters"], 10_843_155)

        full["context_gate"] = 0.0
        self.assertEqual(
            clean_v7_dch_variant_spec(
                "tpd_clean_v7_dch_full",
            )["context_gate"],
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "Unknown Clean-v7 DCH"):
            clean_v7_dch_variant_spec("unknown")

    def test_kcs_representation_is_v6_exact_and_dch_formula_is_correct(
        self,
    ) -> None:
        torch.manual_seed(7)
        v6 = TPDCleanV6Block(
            channels=4,
            activate=False,
            use_context_headroom=True,
        )
        dch = TPDCleanV7DCHBlock(
            channels=4,
            activate=False,
            context_gate=1.0,
        )
        incompatible = dch.load_state_dict(v6.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        with torch.no_grad():
            scale_logits = torch.tensor([0.4, -0.7, 1.2, -1.5])
            v6.saliency_scale.copy_(scale_logits)
            dch.saliency_scale.copy_(scale_logits)

        inputs = torch.randn(2, 4, 12, 16)
        v6_sources = v6.branches(inputs)
        dch_sources = dch.branches(inputs)
        for actual, expected in zip(dch_sources, v6_sources):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            dch.phase_tied_weight(),
            v6.phase_tied_weight(),
            rtol=0.0,
            atol=0.0,
        )

        keep, context, saliency = dch_sources
        tied_weight = dch.phase_tied_weight()
        context_aligned = F.conv2d(
            context.float(),
            tied_weight,
            bias=None,
        )
        saliency_aligned = F.conv2d(
            saliency.float(),
            tied_weight,
            bias=None,
        )
        torch.testing.assert_close(
            dch.context_code(context_aligned),
            v6.context_code(context_aligned),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            dch.context_modulation(context_aligned),
            v6.context_modulation(context_aligned),
            rtol=0.0,
            atol=0.0,
        )

        actual_scale, modulation, headroom = dch.headroom(context_aligned)
        expected_scale = torch.tanh(
            dch.saliency_scale.float(),
        ).view(1, -1, 1, 1)
        expected_headroom = (
            1.0
            + expected_scale.abs()
            * (1.0 - expected_scale.abs())
            * modulation
        )
        torch.testing.assert_close(
            actual_scale,
            expected_scale,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            headroom,
            expected_headroom,
            rtol=0.0,
            atol=0.0,
        )

        actual_keep, residual, actual_saliency, actual_modulation = (
            dch.fusion_terms(inputs)
        )
        expected_residual = (
            saliency_aligned * (expected_scale * expected_headroom)
        ).to(dtype=keep.dtype)
        for actual, expected in (
            (actual_keep, keep),
            (actual_saliency, saliency_aligned),
            (actual_modulation, modulation),
            (residual, expected_residual),
        ):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=0.0,
            )

        modulation_mean = modulation.mean(dim=(-2, -1))
        headroom_mean = headroom.mean(dim=(-2, -1))
        torch.testing.assert_close(
            modulation_mean,
            torch.zeros_like(modulation_mean),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            headroom_mean,
            torch.ones_like(headroom_mean),
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            headroom,
            headroom.clamp(
                min=CONTEXT_HEADROOM_FLOOR,
                max=CONTEXT_HEADROOM_CEILING,
            ),
            rtol=0.0,
            atol=1e-7,
        )
        coefficient = (actual_scale * headroom).abs()
        torch.testing.assert_close(
            coefficient,
            coefficient.clamp(max=1.0),
            rtol=0.0,
            atol=1e-7,
        )
        torch.testing.assert_close(
            residual.abs(),
            torch.minimum(residual.abs(), saliency_aligned.abs()),
            rtol=0.0,
            atol=1e-7,
        )

    def test_capacity_skips_context_path_and_nonzero_full_differs(
        self,
    ) -> None:
        torch.manual_seed(13)
        full = TPDCleanV7DCHBlock(
            channels=4,
            activate=False,
            context_gate=1.0,
        )
        capacity = TPDCleanV7DCHBlock(
            channels=4,
            activate=False,
            context_gate=0.0,
        )
        capacity.load_state_dict(full.state_dict(), strict=True)
        with torch.no_grad():
            scales = torch.tensor([0.3, -0.4, 0.5, -0.6])
            full.saliency_scale.copy_(scales)
            capacity.saliency_scale.copy_(scales)
        inputs = torch.randn(2, 4, 12, 16)

        with (
            mock.patch.object(
                capacity,
                "context_code",
                wraps=capacity.context_code,
            ) as context_code,
            mock.patch.object(
                capacity,
                "context_modulation",
                wraps=capacity.context_modulation,
            ) as context_modulation,
            mock.patch.object(
                capacity,
                "headroom",
                wraps=capacity.headroom,
            ) as headroom,
            mock.patch.object(
                F,
                "conv2d",
                wraps=F.conv2d,
            ) as capacity_conv2d,
            mock.patch.object(
                F,
                "avg_pool2d",
                wraps=F.avg_pool2d,
            ) as capacity_avg_pool,
        ):
            capacity_output = capacity(inputs)
        context_code.assert_not_called()
        context_modulation.assert_not_called()
        headroom.assert_not_called()
        self.assertEqual(capacity_conv2d.call_count, 2)
        self.assertEqual(capacity_avg_pool.call_count, 1)

        with mock.patch.object(
            F,
            "conv2d",
            wraps=F.conv2d,
        ) as full_conv2d:
            full_output = full(inputs)
        self.assertEqual(full_conv2d.call_count, 3)
        self.assertGreater(
            float((full_output - capacity_output).abs().max().detach()),
            0.0,
        )

        _, context, _ = capacity.branches(inputs)
        context_aligned = F.conv2d(
            context.float(),
            capacity.phase_tied_weight(),
            bias=None,
        )
        scale, modulation, capacity_headroom = capacity.headroom(
            context_aligned,
        )
        self.assertTrue(
            torch.equal(modulation, torch.zeros_like(modulation))
        )
        self.assertTrue(
            torch.equal(
                capacity_headroom,
                torch.ones_like(capacity_headroom),
            )
        )
        self.assertTrue(
            torch.equal(
                scale,
                torch.tanh(capacity.saliency_scale.float()).view(
                    1,
                    -1,
                    1,
                    1,
                ),
            )
        )

    def test_zero_saliency_has_exact_zero_residual(self) -> None:
        block = TPDCleanV7DCHBlock(
            channels=2,
            activate=False,
            context_gate=1.0,
        )
        with torch.no_grad():
            block.saliency_scale.copy_(torch.tensor([0.8, -0.7]))
        low_resolution = torch.randn(1, 2, 3, 4)
        inputs = low_resolution.repeat_interleave(
            2,
            dim=-2,
        ).repeat_interleave(2, dim=-1)
        _, _, source_saliency = block.branches(inputs)
        _, residual, saliency_aligned, _ = block.fusion_terms(inputs)
        self.assertTrue(
            torch.equal(source_saliency, torch.zeros_like(source_saliency))
        )
        self.assertTrue(
            torch.equal(
                saliency_aligned,
                torch.zeros_like(saliency_aligned),
            )
        )
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_v6_state_layout_counts_and_five_node_evidence(self) -> None:
        layouts = []
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            embedding1 = build_clean_v7_dch_patch_embedding(
                variant,
                channels=32,
                stride=16,
            )
            embedding2 = build_clean_v7_dch_patch_embedding(
                variant,
                channels=64,
                stride=8,
            )
            self.assertEqual(
                parameter_count(embedding1) + parameter_count(embedding2),
                66_176,
            )
            layouts.append(tuple(
                (prefix + "." + name, tuple(tensor.shape), tensor.dtype)
                for prefix, embedding in (
                    ("embeddings_1", embedding1),
                    ("embeddings_2", embedding2),
                )
                for name, tensor in embedding.state_dict().items()
            ))
            for embedding in (embedding1, embedding2):
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
        dch = build_clean_v7_dch_patch_embedding(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            channels=4,
            stride=8,
        )
        self.assertEqual(tuple(v6.state_dict()), tuple(dch.state_dict()))
        for key in v6.state_dict():
            self.assertEqual(
                v6.state_dict()[key].shape,
                dch.state_dict()[key].shape,
                msg=key,
            )
            self.assertEqual(
                v6.state_dict()[key].dtype,
                dch.state_dict()[key].dtype,
                msg=key,
            )
        incompatible = dch.load_state_dict(v6.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        incompatible = v6.load_state_dict(dch.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

        evidence_cases = (
            (4, 16, (64, 96), 3),
            (8, 8, (32, 48), 2),
        )
        all_nodes = []
        for channels, stride, spatial, expected_nodes in evidence_cases:
            embedding = build_clean_v7_dch_patch_embedding(
                PRIMARY_CLEAN_V7_DCH_VARIANT,
                channels=channels,
                stride=stride,
            )
            inputs = torch.randn(2, channels, *spatial)
            ordinary = embedding(inputs)
            endpoint, nodes = embedding.forward_with_evidence(inputs)
            self.assertTrue(torch.equal(endpoint, ordinary))
            self.assertEqual(len(nodes), expected_nodes)

            manual_states = []
            manual = inputs
            for block in embedding.blocks:
                manual = block(manual)
                manual_states.append(manual)
            self.assertTrue(torch.equal(endpoint, manual_states[-1]))
            for node, reference in zip(nodes, manual_states[:-1]):
                self.assertTrue(torch.equal(node, reference))
            all_nodes.extend(nodes)

            none_endpoint, none_nodes = embedding.forward_with_evidence(None)
            self.assertIsNone(none_endpoint)
            self.assertEqual(none_nodes, ())
            self.assertIsNone(embedding(None))

        self.assertEqual(len(all_nodes), 5)
        self.assertEqual(
            [tuple(node.shape) for node in all_nodes],
            [
                (2, 4, 32, 48),
                (2, 4, 16, 24),
                (2, 4, 8, 12),
                (2, 8, 16, 24),
                (2, 8, 8, 12),
            ],
        )

    def test_full_model_replacement_count_and_dense_spd_zero_anchor(
        self,
    ) -> None:
        seed = 101
        spd = _build_full_spd_model(seed)
        full = _build_full_dch_model(
            "tpd_clean_v7_dch_full",
            seed,
        )
        capacity = _build_full_dch_model(
            "tpd_clean_v7_dch_capacity",
            seed,
        )
        self.assertEqual(parameter_count(full), 10_843_155)
        self.assertEqual(parameter_count(capacity), 10_843_155)
        for model in (full, capacity):
            self.assertIsInstance(
                model.mtc.embeddings_1,
                TPDCleanV7DCHPatchEmbedding,
            )
            self.assertIsInstance(
                model.mtc.embeddings_2,
                TPDCleanV7DCHPatchEmbedding,
            )
            self.assertEqual(len(model.mtc.embeddings_1.blocks), 4)
            self.assertEqual(len(model.mtc.embeddings_2.blocks), 3)
            module_names = tuple(name.lower() for name, _ in model.named_modules())
            self.assertFalse(
                any("relay" in name or "ner" in name for name in module_names)
            )

        inputs = torch.randn(1, 1, 32, 32)
        spd.eval()
        full.eval()
        capacity.eval()
        with torch.inference_mode():
            expected = spd(inputs)
            full_outputs = full(inputs)
            capacity_outputs = capacity(inputs)
        self.assertEqual(len(expected), 6)
        for candidate_outputs in (full_outputs, capacity_outputs):
            self.assertEqual(len(candidate_outputs), 6)
            for actual, reference in zip(candidate_outputs, expected):
                self.assertTrue(torch.equal(actual, reference))

    def test_zero_scale_full_capacity_exact_gradients_and_first_adam(
        self,
    ) -> None:
        full = _build_full_dch_model(
            "tpd_clean_v7_dch_full",
            seed=151,
        )
        capacity = _build_full_dch_model(
            "tpd_clean_v7_dch_capacity",
            seed=151,
        )
        full_state = full.state_dict()
        capacity_state = capacity.state_dict()
        self.assertEqual(tuple(full_state), tuple(capacity_state))
        for key in full_state:
            self.assertTrue(
                torch.equal(full_state[key], capacity_state[key]),
                msg=key,
            )

        optimizer_full = torch.optim.Adam(full.parameters(), lr=1e-3)
        optimizer_capacity = torch.optim.Adam(capacity.parameters(), lr=1e-3)
        self.assert_nested_exact(
            optimizer_full.state_dict(),
            optimizer_capacity.state_dict(),
        )

        input_full = torch.randn(
            1,
            1,
            32,
            32,
            requires_grad=True,
        )
        input_capacity = input_full.detach().clone().requires_grad_(True)
        outputs_full = full(input_full)
        outputs_capacity = capacity(input_capacity)
        self.assertEqual(len(outputs_full), 6)
        for actual, expected in zip(outputs_full, outputs_capacity):
            self.assertTrue(torch.equal(actual, expected))

        loss_full = sum(
            output.square().mean() + output.mean()
            for output in outputs_full
        )
        loss_capacity = sum(
            output.square().mean() + output.mean()
            for output in outputs_capacity
        )
        self.assertTrue(torch.equal(loss_full, loss_capacity))
        loss_full.backward()
        loss_capacity.backward()
        self.assertTrue(
            torch.equal(input_full.grad, input_capacity.grad)
        )

        full_parameters = tuple(full.named_parameters())
        capacity_parameters = tuple(capacity.named_parameters())
        self.assertEqual(
            tuple(name for name, _ in full_parameters),
            tuple(name for name, _ in capacity_parameters),
        )
        for (name, parameter_full), (_, parameter_capacity) in zip(
            full_parameters,
            capacity_parameters,
        ):
            self.assertEqual(
                parameter_full.grad is None,
                parameter_capacity.grad is None,
                msg=f"gradient-presence mismatch: {name}",
            )
            if parameter_full.grad is None:
                continue
            self.assertTrue(
                torch.equal(
                    parameter_full.grad,
                    parameter_capacity.grad,
                ),
                msg=name,
            )

        optimizer_full.step()
        optimizer_capacity.step()
        for key, tensor_full in full.state_dict().items():
            self.assertTrue(
                torch.equal(
                    tensor_full,
                    capacity.state_dict()[key],
                ),
                msg=key,
            )
        self.assert_nested_exact(
            optimizer_full.state_dict(),
            optimizer_capacity.state_dict(),
        )

    def test_strict_save_reload_shapes_and_invalid_contracts(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            embedding = build_clean_v7_dch_patch_embedding(
                variant,
                channels=4,
                stride=4,
            )
            inputs = torch.randn(2, 4, 16, 20)
            expected = embedding(inputs)
            state_buffer = io.BytesIO()
            torch.save(embedding.state_dict(), state_buffer)
            state_buffer.seek(0)
            rebuilt = build_clean_v7_dch_patch_embedding(
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
            self.assertTrue(torch.equal(rebuilt(inputs), expected))

        embedding = build_clean_v7_dch_patch_embedding(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            channels=4,
            stride=4,
        )
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
            build_clean_v7_dch_patch_embedding(
                PRIMARY_CLEAN_V7_DCH_VARIANT,
                channels=4,
                stride=6,
            )
        with self.assertRaisesRegex(ValueError, "channels must be positive"):
            TPDCleanV7DCHBlock(
                channels=0,
                activate=False,
                context_gate=1.0,
            )
        with self.assertRaisesRegex(ValueError, "context_gate"):
            TPDCleanV7DCHBlock(
                channels=4,
                activate=False,
                context_gate=0.5,
            )
        with self.assertRaisesRegex(ValueError, "eps must be positive"):
            TPDCleanV7DCHBlock(
                channels=4,
                activate=False,
                context_gate=1.0,
                eps=0.0,
            )

    def test_zero_scale_patch_is_exact_dense_spd(self) -> None:
        for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS:
            torch.manual_seed(17)
            spd = SPDPatchEmbedding(channels=4, stride=8)
            torch.manual_seed(17)
            dch = build_clean_v7_dch_patch_embedding(
                variant,
                channels=4,
                stride=8,
            )
            torch.manual_seed(29)
            spd.apply(weights_init_kaiming)
            torch.manual_seed(29)
            dch.apply(weights_init_kaiming)

            spd_x = torch.randn(2, 4, 32, 40)
            dch_x = spd_x.clone()
            for spd_block, dch_block in zip(spd.blocks, dch.blocks):
                self.assertEqual(
                    int(torch.count_nonzero(dch_block.saliency_scale)),
                    0,
                )
                self.assertTrue(
                    torch.equal(
                        dch_block.phase_compress.weight,
                        spd_block.phase_compress.weight,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dch_block.phase_compress.bias,
                        spd_block.phase_compress.bias,
                    )
                )
                spd_x = spd_block(spd_x)
                dch_x = dch_block(dch_x)
                self.assertTrue(torch.equal(dch_x, spd_x))


if __name__ == "__main__":
    unittest.main()
