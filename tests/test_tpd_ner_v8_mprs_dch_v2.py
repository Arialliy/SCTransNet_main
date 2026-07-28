from __future__ import annotations

import gc
import inspect
import math
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.train_tpd_clean_v8_mprs_dch import (
    TOTAL_PARAMETERS,
    build_clean_v8_mprs_dch_model,
)
from experiments.train_tpd_pilot import weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    TPDCleanV8MPRSDCHBlock,
    TPDCleanV8MPRSDCHPatchEmbedding,
    build_clean_v8_mprs_dch_patch_embedding,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    PRODUCTION_PARENT_PARAMETERS,
    TPDNERV8MPRSDCHSCTransNet,
    adapt_v8_mprs_dch_parent,
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    DEFAULT_RELAY_WIDTH,
    PRODUCTION_V2_RELAY_ON_PARAMETERS,
    PRODUCTION_V2_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    RMSBalancedCenteredEvidenceRelay,
    RMSBalancedRelayFusionCell,
    TPDNERV8MPRSDCHV2SCTransNet,
    V2_MASK_LIMIT,
    V2_SKIP_FACTOR_BOUNDS,
    adapt_v8_mprs_dch_parent_v2,
    arctangent_residual_mask,
    sample_full_tensor_rms_normalize,
    spatially_center_gate_logits,
    v2_relay_parameter_count,
)
from model.tpd_sctransnet import ExplicitNestedEvidenceRelay


FULL = "tpd_clean_v8_mprs_dch_full"
CAPACITY = "tpd_clean_v8_mprs_dch_capacity"

torch.set_num_threads(1)


def _small_config():
    config = get_SCTrans_config()
    config.base_channel = 4
    config.KV_size = 60
    config.transformer.num_layers = 1
    return config


def _small_parent(
    variant: str = FULL,
    *,
    seed: int = 42,
) -> SCTransNet:
    torch.manual_seed(seed)
    model = SCTransNet(
        _small_config(),
        img_size=32,
        mode="train",
        deepsuper=True,
    )
    model.apply(weights_init_kaiming)
    replacements = {
        "embeddings_1": build_clean_v8_mprs_dch_patch_embedding(
            variant,
            channels=4,
            stride=16,
        ),
        "embeddings_2": build_clean_v8_mprs_dch_patch_embedding(
            variant,
            channels=8,
            stride=8,
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    for replacement in replacements.values():
        replacement.apply(weights_init_kaiming)
    return model


def _adapt_pair(
    parent: SCTransNet,
    variant: str = FULL,
) -> tuple[
    TPDNERV8MPRSDCHSCTransNet,
    TPDNERV8MPRSDCHV2SCTransNet,
]:
    off = adapt_v8_mprs_dch_parent_v2(
        parent,
        variant=variant,
        relay_enabled=False,
    )
    on = adapt_v8_mprs_dch_parent_v2(
        parent,
        variant=variant,
        relay_enabled=True,
    )
    if not isinstance(on, TPDNERV8MPRSDCHV2SCTransNet):
        raise TypeError("V2 relay-on fixture has the wrong model type")
    return off, on


def _six_output_loss(
    outputs: object,
    targets: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("expected six deep-supervision outputs")
    criterion = nn.BCELoss(reduction="mean")
    return sum(criterion(output, targets) for output in outputs)


def _gradient_l1(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("expected a relay gradient tensor")
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("relay gradient is non-finite")
        total += float(parameter.grad.detach().abs().sum())
    return total


def _reference_rms(
    tensor: torch.Tensor,
    eps: float = RELAY_RMS_EPS,
) -> torch.Tensor:
    working = (
        tensor.float()
        if tensor.dtype in (torch.float16, torch.bfloat16)
        else tensor
    )
    normalized = working * torch.rsqrt(
        working.square().mean(dim=(1, 2, 3), keepdim=True) + eps
    )
    return normalized.to(tensor.dtype)


class V2RelayPrimitiveTests(unittest.TestCase):
    def test_double_rms_matches_formula_and_zero_is_finite(self) -> None:
        torch.manual_seed(1401)
        cell = RMSBalancedRelayFusionCell((2, 3), width=8)
        source1 = torch.randn(2, 2, 8, 8)
        source2 = torch.randn(2, 3, 4, 4)
        captured: list[torch.Tensor] = []

        def capture_fuse_input(_module, inputs):
            captured.append(inputs[0].detach().clone())

        handle = cell.fuse.register_forward_pre_hook(capture_fuse_input)
        try:
            output = cell((source1, source2), (8, 8))
        finally:
            handle.remove()

        expected_sources = []
        for source, projection in zip(
            (source1, source2),
            cell.projections,
        ):
            value = projection(source)
            if value.shape[-2:] != (8, 8):
                value = F.interpolate(
                    value,
                    size=(8, 8),
                    mode="bilinear",
                    align_corners=False,
                )
            expected_sources.append(_reference_rms(value))
        expected_fuse_input = torch.cat(expected_sources, dim=1)
        self.assertEqual(len(captured), 1)
        self.assertTrue(
            torch.allclose(
                captured[0],
                expected_fuse_input,
                atol=1e-7,
                rtol=1e-6,
            )
        )

        expected_fused = F.relu(
            cell.fuse(expected_fuse_input),
            inplace=False,
        )
        expected_output = _reference_rms(expected_fused)
        self.assertTrue(
            torch.allclose(output, expected_output, atol=1e-7, rtol=1e-6)
        )
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(bool((output >= 0).all()))

        scaled1 = torch.cat((source1[:1], 7.0 * source1[:1]), dim=0)
        scaled2 = torch.cat((source2[:1], 7.0 * source2[:1]), dim=0)
        scaled_output = cell((scaled1, scaled2), (8, 8))
        self.assertTrue(
            torch.allclose(
                scaled_output[:1],
                scaled_output[1:],
                atol=5e-6,
                rtol=5e-6,
            )
        )

        zero_output = cell(
            (
                torch.zeros(2, 2, 8, 8),
                torch.zeros(2, 3, 4, 4),
            ),
            (8, 8),
        )
        self.assertTrue(torch.isfinite(zero_output).all())
        self.assertEqual(int(torch.count_nonzero(zero_output)), 0)
        direct_zero = sample_full_tensor_rms_normalize(
            torch.zeros(3, 4, 5, 6)
        )
        self.assertTrue(torch.isfinite(direct_zero).all())
        self.assertEqual(int(torch.count_nonzero(direct_zero)), 0)

    def test_centered_arctangent_formula_bounds_offset_and_derivative(self) -> None:
        torch.manual_seed(1402)
        relay = RMSBalancedCenteredEvidenceRelay(base_channels=2)
        gate = relay.gates["4"]
        self.assertIsNone(gate.bias)
        with torch.no_grad():
            gate.weight.copy_(
                torch.linspace(-0.75, 0.75, steps=8).view(1, 8, 1, 1)
            )
        sources = (
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 16, 8, 8),
        )
        relay_value, mask = relay.forward_stage(4, sources, (8, 8))
        logits = gate(relay_value)
        expected_centered = logits.float() - logits.float().mean(
            dim=(-2, -1),
            keepdim=True,
        )
        expected_mask = torch.atan(math.pi * expected_centered) / math.pi
        self.assertTrue(
            torch.allclose(mask, expected_mask, atol=1e-7, rtol=1e-6)
        )
        self.assertEqual(mask.dtype, relay_value.dtype)
        self.assertEqual(mask.device, relay_value.device)
        self.assertTrue(torch.isfinite(mask).all())
        bounded = mask.detach()
        self.assertGreater(float(bounded.min()), -V2_MASK_LIMIT)
        self.assertLess(float(bounded.max()), V2_MASK_LIMIT)
        self.assertGreater(
            float((1.0 + bounded).min()),
            V2_SKIP_FACTOR_BOUNDS[0],
        )
        self.assertLess(
            float((1.0 + bounded).max()),
            V2_SKIP_FACTOR_BOUNDS[1],
        )

        offsets = torch.tensor([3.0, -5.0]).view(2, 1, 1, 1)
        shifted = arctangent_residual_mask(
            spatially_center_gate_logits(logits + offsets)
        )
        self.assertTrue(torch.allclose(mask, shifted, atol=1e-6, rtol=1e-6))

        zero = torch.zeros((), dtype=torch.float64, requires_grad=True)
        mapped = arctangent_residual_mask(zero)
        derivative = torch.autograd.grad(mapped, zero)[0]
        self.assertEqual(float(mapped.detach()), 0.0)
        self.assertTrue(
            torch.allclose(
                derivative,
                torch.ones_like(derivative),
                atol=1e-12,
                rtol=0.0,
            )
        )

        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            with self.subTest(dtype=dtype):
                maximum = torch.finfo(dtype).max
                extreme = torch.tensor((-maximum, maximum), dtype=dtype)
                extreme_mask = arctangent_residual_mask(extreme)
                one = torch.tensor(1.0, dtype=dtype)
                lower_factor = torch.nextafter(
                    torch.tensor(V2_SKIP_FACTOR_BOUNDS[0], dtype=dtype),
                    one,
                )
                upper_factor = torch.nextafter(
                    torch.tensor(V2_SKIP_FACTOR_BOUNDS[1], dtype=dtype),
                    one,
                )
                lower_mask = lower_factor - one
                upper_mask = upper_factor - one
                extreme_factor = one + extreme_mask
                self.assertTrue(torch.isfinite(extreme_mask).all())
                self.assertEqual(float(extreme_mask[0]), float(lower_mask))
                self.assertEqual(float(extreme_mask[1]), float(upper_mask))
                self.assertEqual(float(extreme_factor[0]), float(lower_factor))
                self.assertEqual(float(extreme_factor[1]), float(upper_factor))
                self.assertGreater(
                    float(extreme_factor.min()),
                    V2_SKIP_FACTOR_BOUNDS[0],
                )
                self.assertLess(
                    float(extreme_factor.max()),
                    V2_SKIP_FACTOR_BOUNDS[1],
                )

        for nonfinite in (float("nan"), float("inf"), -float("inf")):
            with (
                self.subTest(nonfinite=nonfinite),
                self.assertRaisesRegex(FloatingPointError, "must be finite"),
            ):
                arctangent_residual_mask(torch.tensor(nonfinite))


class V2ModelContractTests(unittest.TestCase):
    def test_structure_prefix_parent_state_and_relay_off_identity(self) -> None:
        for variant in (FULL, CAPACITY):
            with self.subTest(variant=variant):
                parent = _small_parent(variant)
                with torch.no_grad():
                    value = 0.03125
                    for embedding_name in ("embeddings_1", "embeddings_2"):
                        for block in getattr(parent.mtc, embedding_name).blocks:
                            block.saliency_scale.fill_(value)
                            value += 0.03125
                parent_before = {
                    name: tensor.detach().clone()
                    for name, tensor in parent.state_dict().items()
                }

                v1_off = adapt_v8_mprs_dch_parent(
                    parent,
                    variant=variant,
                    relay_enabled=False,
                )
                v2_off, v2_on = _adapt_pair(parent, variant)
                self.assertIs(
                    type(v2_off),
                    TPDNERV8MPRSDCHSCTransNet,
                )
                self.assertNotIsInstance(v2_off, TPDNERV8MPRSDCHV2SCTransNet)
                self.assertIsInstance(v2_on, TPDNERV8MPRSDCHV2SCTransNet)
                self.assertFalse(v2_off.relay_enabled)
                self.assertFalse(hasattr(v2_off, "tpd_ner"))
                self.assertTrue(v2_on.relay_enabled)
                self.assertEqual(v2_on.relay_width, DEFAULT_RELAY_WIDTH)
                self.assertEqual(
                    v2_on.relay_initialization_seed,
                    DEFAULT_RELAY_INITIALIZATION_SEED,
                )

                for name, value in parent_before.items():
                    self.assertTrue(
                        torch.equal(value, parent.state_dict()[name]),
                        f"parent changed: {name}",
                    )
                    self.assertTrue(
                        torch.equal(value, v2_off.state_dict()[name]),
                        f"relay-off parent differs: {name}",
                    )
                    self.assertTrue(
                        torch.equal(value, v2_on.state_dict()[name]),
                        f"relay-on parent differs: {name}",
                    )
                for name, value in v1_off.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, v2_off.state_dict()[name]),
                        f"V2 relay-off identity differs: {name}",
                    )

                off_state = v2_off.state_dict()
                on_state = v2_on.state_dict()
                added = set(on_state) - set(off_state)
                self.assertEqual(len(added), 16)
                self.assertTrue(
                    all(name.startswith("tpd_ner.") for name in added)
                )
                self.assertFalse(set(off_state) - set(on_state))
                for name, value in off_state.items():
                    self.assertTrue(torch.equal(value, on_state[name]), name)
                for gate in v2_on.tpd_ner.gates.values():
                    self.assertIsNone(gate.bias)
                    self.assertEqual(int(torch.count_nonzero(gate.weight)), 0)

                manifest = v2_on.architecture_manifest()
                self.assertEqual(manifest["evidence_layout"], (3, 2))
                self.assertEqual(manifest["evidence_node_count"], 5)
                self.assertEqual(manifest["relay_stage_order"], (4, 3, 2))
                self.assertEqual(manifest["relay_width"], 8)
                self.assertEqual(
                    manifest["semantic_sources"],
                    ("Keep", "Context", "Saliency"),
                )
                self.assertEqual(
                    manifest["relay_version"],
                    "v2_rms_centered_arctangent",
                )
                self.assertTrue(
                    manifest["source_projection_rms_normalized"]
                )
                self.assertTrue(
                    manifest["fusion_relu_output_rms_normalized"]
                )
                self.assertEqual(manifest["relay_rms_eps"], RELAY_RMS_EPS)
                self.assertFalse(manifest["gate_bias"])
                self.assertFalse(manifest["fourth_parallel_branch_added"])
                del parent, v1_off, v2_off, v2_on
                gc.collect()

    def test_production_parameters_cpu256_and_strict_checkpoint_isolation(
        self,
    ) -> None:
        self.assertEqual(PRODUCTION_PARENT_PARAMETERS, TOTAL_PARAMETERS)
        parent, _ = build_clean_v8_mprs_dch_model(FULL, seed=42)
        off, on = _adapt_pair(parent, FULL)
        self.assertEqual(
            sum(parameter.numel() for parameter in off.parameters()),
            PRODUCTION_PARENT_PARAMETERS,
        )
        self.assertEqual(
            v2_relay_parameter_count(on),
            PRODUCTION_V2_RELAY_PARAMETERS,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in on.parameters()),
            PRODUCTION_V2_RELAY_ON_PARAMETERS,
        )
        self.assertEqual(v2_relay_parameter_count(off), 0)

        on.eval()
        with torch.no_grad():
            outputs = on(torch.randn(1, 1, 256, 256))
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 6)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
            self.assertTrue(torch.isfinite(output).all())

        rebuilt = adapt_v8_mprs_dch_parent_v2(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        incompatible = rebuilt.load_state_dict(on.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)

        v1_on = adapt_v8_mprs_dch_parent(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        with self.assertRaises(RuntimeError):
            rebuilt.load_state_dict(v1_on.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            v1_on.load_state_dict(rebuilt.state_dict(), strict=True)
        del parent, off, on, rebuilt, v1_on
        gc.collect()

    def test_five_node_geometry_and_initial_six_output_identity(self) -> None:
        parent = _small_parent()
        off, on = _adapt_pair(parent)
        inputs = torch.randn(2, 1, 32, 32)

        on.eval()
        with torch.no_grad():
            x1 = on.inc(inputs)
            x2 = on.down_encoder1(on.pool(x1))
            x3 = on.down_encoder2(on.pool(x2))
            x4 = on.down_encoder3(on.pool(x3))
            emb1, emb2, _, _, evidence1, evidence2 = (
                on.explicit_embeddings(x1, x2, x3, x4)
            )
        self.assertEqual(
            tuple(tuple(node.shape) for node in evidence1),
            (
                (2, 4, 16, 16),
                (2, 4, 8, 8),
                (2, 4, 4, 4),
            ),
        )
        self.assertEqual(
            tuple(tuple(node.shape) for node in evidence2),
            (
                (2, 8, 8, 8),
                (2, 8, 4, 4),
            ),
        )
        self.assertEqual(tuple(emb1.shape), (2, 4, 2, 2))
        self.assertEqual(tuple(emb2.shape), (2, 8, 2, 2))

        off.eval()
        with torch.no_grad():
            off_outputs = off(inputs)
            on_outputs = on(inputs)
        self.assertEqual(len(off_outputs), 6)
        self.assertEqual(len(on_outputs), 6)
        for index, (off_output, on_output) in enumerate(
            zip(off_outputs, on_outputs)
        ):
            self.assertEqual(tuple(on_output.shape), (2, 1, 32, 32))
            self.assertTrue(
                torch.equal(off_output, on_output),
                f"step-zero output {index}",
            )

    def test_zero_gate_first_adam_identity_and_two_step_gradient(self) -> None:
        parent = _small_parent(seed=73)
        off, on = _adapt_pair(parent)
        off.train()
        on.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7301)
        inputs = torch.randn(2, 1, 32, 32, generator=generator)
        targets = torch.rand(2, 1, 32, 32, generator=generator)
        off_optimizer = torch.optim.Adam(off.parameters(), lr=1e-3)
        on_optimizer = torch.optim.Adam(on.parameters(), lr=1e-3)

        off_optimizer.zero_grad(set_to_none=True)
        on_optimizer.zero_grad(set_to_none=True)
        off_outputs = off(inputs)
        on_outputs = on(inputs)
        for index, (off_output, on_output) in enumerate(
            zip(off_outputs, on_outputs)
        ):
            self.assertTrue(
                torch.equal(off_output, on_output),
                f"pre-Adam output {index}",
            )
        off_loss = _six_output_loss(off_outputs, targets)
        on_loss = _six_output_loss(on_outputs, targets)
        self.assertTrue(torch.equal(off_loss, on_loss))
        off_loss.backward()
        on_loss.backward()

        off_parameters = dict(off.named_parameters())
        on_parameters = dict(on.named_parameters())
        for name, off_parameter in off_parameters.items():
            on_parameter = on_parameters[name]
            if off_parameter.grad is None:
                self.assertIsNone(on_parameter.grad, name)
            else:
                self.assertIsNotNone(on_parameter.grad, name)
                self.assertTrue(
                    torch.equal(off_parameter.grad, on_parameter.grad),
                    f"shared gradient {name}",
                )
        gate_gradient = _gradient_l1(on.tpd_ner.gates.parameters())
        fusion_gradient = _gradient_l1(on.tpd_ner.fusions.parameters())
        self.assertGreater(gate_gradient, 0.0)
        self.assertEqual(fusion_gradient, 0.0)

        off_optimizer.step()
        on_optimizer.step()
        off_state = off.state_dict()
        on_state = on.state_dict()
        for name, value in off_state.items():
            self.assertTrue(
                torch.equal(value, on_state[name]),
                f"first-Adam shared state {name}",
            )
        for name, off_parameter in off_parameters.items():
            on_parameter = on_parameters[name]
            off_adam = off_optimizer.state.get(off_parameter, {})
            on_adam = on_optimizer.state.get(on_parameter, {})
            self.assertEqual(set(off_adam), set(on_adam), name)
            for key, off_value in off_adam.items():
                on_value = on_adam[key]
                if isinstance(off_value, torch.Tensor):
                    self.assertTrue(
                        torch.equal(off_value, on_value),
                        f"Adam {name}.{key}",
                    )
                else:
                    self.assertEqual(off_value, on_value)

        on_optimizer.zero_grad(set_to_none=True)
        second_loss = _six_output_loss(on(inputs), targets)
        second_loss.backward()
        self.assertTrue(torch.isfinite(second_loss))
        self.assertGreater(
            _gradient_l1(on.tpd_ner.fusions.parameters()),
            0.0,
        )

    def test_recursive_wiring_single_compute_and_no_persistent_tensor_cache(
        self,
    ) -> None:
        parent = _small_parent()
        _, model = _adapt_pair(parent)
        model.eval()

        block_calls: dict[int, int] = {}
        embedding_calls: dict[int, int] = {}
        fusion_calls: dict[int, int] = {}
        stages: list[tuple[int, tuple[int, ...], int]] = []
        real_block_forward = TPDCleanV8MPRSDCHBlock.forward
        real_evidence_forward = (
            TPDCleanV8MPRSDCHPatchEmbedding.forward_with_evidence
        )
        real_fusion_forward = RMSBalancedRelayFusionCell.forward
        real_stage_forward = RMSBalancedCenteredEvidenceRelay.forward_stage

        def counted_block_forward(block, inputs):
            block_calls[id(block)] = block_calls.get(id(block), 0) + 1
            return real_block_forward(block, inputs)

        def counted_evidence_forward(embedding, inputs):
            embedding_calls[id(embedding)] = (
                embedding_calls.get(id(embedding), 0) + 1
            )
            return real_evidence_forward(embedding, inputs)

        def counted_fusion_forward(cell, sources, output_size):
            fusion_calls[id(cell)] = fusion_calls.get(id(cell), 0) + 1
            return real_fusion_forward(cell, sources, output_size)

        def counted_stage_forward(relay, stage, sources, output_size):
            relay_value, mask = real_stage_forward(
                relay,
                stage,
                sources,
                output_size,
            )
            stages.append(
                (stage, tuple(id(source) for source in sources), id(relay_value))
            )
            return relay_value, mask

        def forbidden_mtc_forward(*_args, **_kwargs):
            raise RuntimeError("V2 relay forward must not call mtc.forward")

        def forbidden_diagnostics(*_args, **_kwargs):
            raise RuntimeError("V2 ordinary forward must not call diagnostics")

        def forbidden_v1_relay(*_args, **_kwargs):
            raise RuntimeError("V2 must not execute the V1 relay")

        with (
            mock.patch.object(
                TPDCleanV8MPRSDCHBlock,
                "forward",
                new=counted_block_forward,
            ),
            mock.patch.object(
                TPDCleanV8MPRSDCHPatchEmbedding,
                "forward_with_evidence",
                new=counted_evidence_forward,
            ),
            mock.patch.object(
                RMSBalancedRelayFusionCell,
                "forward",
                new=counted_fusion_forward,
            ),
            mock.patch.object(
                RMSBalancedCenteredEvidenceRelay,
                "forward_stage",
                new=counted_stage_forward,
            ),
            mock.patch.object(
                type(model.mtc),
                "forward",
                new=forbidden_mtc_forward,
            ),
            mock.patch.object(
                TPDCleanV8MPRSDCHBlock,
                "forward_with_mprs_diagnostics",
                new=forbidden_diagnostics,
            ),
            mock.patch.object(
                ExplicitNestedEvidenceRelay,
                "forward_stage",
                new=forbidden_v1_relay,
            ),
        ):
            with torch.no_grad():
                outputs = model(torch.randn(1, 1, 32, 32))

        self.assertEqual(len(outputs), 6)
        self.assertEqual(len(block_calls), 7)
        self.assertTrue(all(count == 1 for count in block_calls.values()))
        self.assertEqual(len(embedding_calls), 2)
        self.assertTrue(all(count == 1 for count in embedding_calls.values()))
        self.assertEqual(len(fusion_calls), 3)
        self.assertTrue(all(count == 1 for count in fusion_calls.values()))
        self.assertEqual([entry[0] for entry in stages], [4, 3, 2])
        self.assertEqual([len(entry[1]) for entry in stages], [3, 4, 3])
        self.assertEqual(stages[0][2], stages[1][1][2])
        self.assertEqual(stages[1][2], stages[2][1][1])

        for module in (
            model.tpd_ner,
            *model.tpd_ner.fusions.values(),
        ):
            tensor_attributes = [
                name
                for name, value in vars(module).items()
                if isinstance(value, torch.Tensor)
            ]
            self.assertFalse(tensor_attributes, tensor_attributes)

        inspected_sources = "\n".join(
            (
                inspect.getsource(RMSBalancedRelayFusionCell.forward),
                inspect.getsource(
                    RMSBalancedCenteredEvidenceRelay.forward_stage
                ),
                inspect.getsource(TPDNERV8MPRSDCHV2SCTransNet),
            )
        )
        for forbidden in (
            ".detach(",
            ".clone(",
            "register_forward_hook",
            "register_forward_pre_hook",
            "forward_with_mprs_diagnostics",
            "mtc.forward(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, inspected_sources)

    def test_width_is_fixed_and_variant_identity_remains_checked(self) -> None:
        parent = _small_parent(FULL)
        with self.assertRaisesRegex(ValueError, "fixed to 8"):
            adapt_v8_mprs_dch_parent_v2(
                parent,
                variant=FULL,
                relay_enabled=True,
                relay_width=2,
            )
        with self.assertRaisesRegex(ValueError, "context gate"):
            adapt_v8_mprs_dch_parent_v2(
                parent,
                variant=CAPACITY,
                relay_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
