from __future__ import annotations

import gc
import math
import unittest

import torch
import torch.nn as nn

from experiments.train_tpd_clean_v8_mprs_dch import (
    TOTAL_PARAMETERS,
    build_clean_v8_mprs_dch_model,
)
from experiments.train_tpd_pilot import weights_init_kaiming
from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    build_clean_v8_mprs_dch_patch_embedding,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    PRODUCTION_PARENT_PARAMETERS,
    TPDNERV8MPRSDCHSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v2 import (
    PRODUCTION_V2_RELAY_PARAMETERS,
    TPDNERV8MPRSDCHV2SCTransNet,
    V2_MASK_LIMIT,
    V2_SKIP_FACTOR_BOUNDS,
    adapt_v8_mprs_dch_parent_v2,
    spatially_center_gate_logits,
)
from model.tpd_ner_v8_mprs_dch_v3 import (
    PRODUCTION_V3_RELAY_ON_PARAMETERS,
    PRODUCTION_V3_RELAY_PARAMETERS,
    RMSBalancedCenteredDCOffsetEvidenceRelay,
    TPDNERV8MPRSDCHV3SCTransNet,
    V3_RELAY_VERSION,
    adapt_v8_mprs_dch_parent_v3,
    v3_relay_parameter_count,
)


FULL = "tpd_clean_v8_mprs_dch_full"
CAPACITY = "tpd_clean_v8_mprs_dch_capacity"
OFFSET_KEYS = {
    "tpd_ner.dc_offsets.4",
    "tpd_ner.dc_offsets.3",
    "tpd_ner.dc_offsets.2",
}

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


def _six_output_loss(
    outputs: object,
    target: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("expected exactly six deep-supervision outputs")
    criterion = nn.BCELoss(reduction="mean")
    return sum(criterion(output, target) for output in outputs)


class V3RelayPrimitiveTests(unittest.TestCase):
    def test_offset_keys_zero_reset_formula_sign_monotonic_and_bounds(
        self,
    ) -> None:
        torch.manual_seed(3101)
        relay = RMSBalancedCenteredDCOffsetEvidenceRelay(
            base_channels=2,
        )
        self.assertEqual(set(relay.dc_offsets), {"4", "3", "2"})
        for stage in ("4", "3", "2"):
            self.assertEqual(tuple(relay.dc_offsets[stage].shape), (1,))
            self.assertEqual(
                int(torch.count_nonzero(relay.dc_offsets[stage])),
                0,
            )
            self.assertIsNone(relay.gates[stage].bias)
            self.assertEqual(
                int(torch.count_nonzero(relay.gates[stage].weight)),
                0,
            )

        gate = relay.gates["4"]
        with torch.no_grad():
            gate.weight.copy_(
                torch.linspace(-0.5, 0.5, steps=8).view(1, 8, 1, 1)
            )
        sources = (
            torch.randn(2, 2, 8, 8),
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 16, 8, 8),
        )
        masks = {}
        relay_values = {}
        for label, offset in (
            ("negative", -0.25),
            ("zero", 0.0),
            ("positive", 0.25),
        ):
            with torch.no_grad():
                relay.dc_offsets["4"].fill_(offset)
            relay_value, mask = relay.forward_stage(4, sources, (8, 8))
            relay_values[label] = relay_value
            masks[label] = mask
            logits = gate(relay_value)
            centered = spatially_center_gate_logits(logits)
            shifted = centered + offset
            expected = torch.atan(math.pi * shifted) / math.pi
            self.assertTrue(
                torch.allclose(mask, expected, atol=1e-7, rtol=1e-6)
            )
            self.assertTrue(torch.isfinite(mask).all())
            self.assertGreater(
                float((1.0 + mask).min().detach()),
                V2_SKIP_FACTOR_BOUNDS[0],
            )
            self.assertLess(
                float((1.0 + mask).max().detach()),
                V2_SKIP_FACTOR_BOUNDS[1],
            )

        self.assertTrue(
            torch.equal(relay_values["negative"], relay_values["zero"])
        )
        self.assertTrue(
            torch.equal(relay_values["zero"], relay_values["positive"])
        )
        self.assertTrue(bool((masks["negative"] < masks["zero"]).all()))
        self.assertTrue(bool((masks["zero"] < masks["positive"]).all()))

        with torch.no_grad():
            gate.weight.zero_()
            relay.dc_offsets["4"].fill_(-0.25)
        _, negative = relay.forward_stage(4, sources, (8, 8))
        with torch.no_grad():
            relay.dc_offsets["4"].fill_(0.25)
        _, positive = relay.forward_stage(4, sources, (8, 8))
        self.assertTrue(bool((negative < 0).all()))
        self.assertTrue(bool((positive > 0).all()))

        with torch.no_grad():
            relay.dc_offsets["4"].fill_(torch.finfo(torch.float32).max)
        _, upper = relay.forward_stage(4, sources, (8, 8))
        with torch.no_grad():
            relay.dc_offsets["4"].fill_(-torch.finfo(torch.float32).max)
        _, lower = relay.forward_stage(4, sources, (8, 8))
        self.assertTrue(torch.isfinite(upper).all())
        self.assertTrue(torch.isfinite(lower).all())
        self.assertLess(float(upper.max().detach()), V2_MASK_LIMIT)
        self.assertGreater(float(lower.min().detach()), -V2_MASK_LIMIT)

        with torch.no_grad():
            for stage in ("4", "3", "2"):
                relay.gates[stage].weight.fill_(1.0)
                relay.dc_offsets[stage].fill_(1.0)
        relay.zero_init_gates()
        for stage in ("4", "3", "2"):
            self.assertEqual(
                int(torch.count_nonzero(relay.gates[stage].weight)),
                0,
            )
            self.assertEqual(
                int(torch.count_nonzero(relay.dc_offsets[stage])),
                0,
            )

    def test_all_three_offsets_receive_finite_nonzero_gradients(self) -> None:
        torch.manual_seed(3102)
        relay = RMSBalancedCenteredDCOffsetEvidenceRelay(
            base_channels=2,
        )
        stage_sources = {
            4: (
                torch.randn(2, 2, 8, 8),
                torch.randn(2, 4, 8, 8),
                torch.randn(2, 16, 8, 8),
            ),
            3: (
                torch.randn(2, 2, 8, 8),
                torch.randn(2, 4, 8, 8),
                torch.randn(2, 8, 8, 8),
                torch.randn(2, 8, 8, 8),
            ),
            2: (
                torch.randn(2, 2, 8, 8),
                torch.randn(2, 8, 8, 8),
                torch.randn(2, 4, 8, 8),
            ),
        }
        loss = torch.zeros(())
        for stage in (4, 3, 2):
            _, mask = relay.forward_stage(
                stage,
                stage_sources[stage],
                (8, 8),
            )
            loss = loss + mask.mean()
        loss.backward()
        for stage in ("4", "3", "2"):
            gradient = relay.dc_offsets[stage].grad
            self.assertIsNotNone(gradient)
            assert gradient is not None
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


class V3ModelContractTests(unittest.TestCase):
    def test_parent_state_topology_manifest_and_v2_shared_initialization(
        self,
    ) -> None:
        for variant in (FULL, CAPACITY):
            with self.subTest(variant=variant):
                parent = _small_parent(variant)
                parent_before = {
                    name: value.detach().clone()
                    for name, value in parent.state_dict().items()
                }
                off = adapt_v8_mprs_dch_parent_v3(
                    parent,
                    variant=variant,
                    relay_enabled=False,
                )
                v2 = adapt_v8_mprs_dch_parent_v2(
                    parent,
                    variant=variant,
                    relay_enabled=True,
                )
                v3 = adapt_v8_mprs_dch_parent_v3(
                    parent,
                    variant=variant,
                    relay_enabled=True,
                )
                self.assertIs(type(off), TPDNERV8MPRSDCHSCTransNet)
                self.assertIsInstance(v2, TPDNERV8MPRSDCHV2SCTransNet)
                self.assertNotIsInstance(v2, TPDNERV8MPRSDCHV3SCTransNet)
                self.assertIsInstance(v3, TPDNERV8MPRSDCHV3SCTransNet)
                self.assertEqual(
                    v3.relay_width,
                    DEFAULT_RELAY_WIDTH,
                )
                self.assertEqual(
                    v3.relay_initialization_seed,
                    DEFAULT_RELAY_INITIALIZATION_SEED,
                )

                for name, value in parent_before.items():
                    self.assertTrue(
                        torch.equal(value, parent.state_dict()[name]),
                        f"parent changed: {name}",
                    )
                    self.assertTrue(
                        torch.equal(value, off.state_dict()[name]),
                        f"relay-off parent differs: {name}",
                    )
                    self.assertTrue(
                        torch.equal(value, v3.state_dict()[name]),
                        f"V3 parent differs: {name}",
                    )

                v2_state = v2.state_dict()
                v3_state = v3.state_dict()
                self.assertEqual(set(v3_state) - set(v2_state), OFFSET_KEYS)
                self.assertFalse(set(v2_state) - set(v3_state))
                for name, value in v2_state.items():
                    self.assertTrue(
                        torch.equal(value, v3_state[name]),
                        f"V2/V3 shared initialization differs: {name}",
                    )
                for name in OFFSET_KEYS:
                    self.assertEqual(
                        int(torch.count_nonzero(v3_state[name])),
                        0,
                    )

                manifest = v3.architecture_manifest()
                self.assertEqual(
                    manifest["relay_version"],
                    V3_RELAY_VERSION,
                )
                self.assertEqual(manifest["evidence_node_count"], 5)
                self.assertEqual(manifest["evidence_layout"], (3, 2))
                self.assertEqual(manifest["relay_stage_order"], (4, 3, 2))
                self.assertEqual(
                    manifest["semantic_sources"],
                    ("Keep", "Context", "Saliency"),
                )
                self.assertFalse(manifest["fourth_parallel_branch_added"])
                self.assertFalse(manifest["gate_bias"])
                self.assertEqual(
                    manifest["gate_spatial_centering"],
                    "per_sample_mean_hw",
                )
                self.assertEqual(
                    manifest["gate_dc_offset"],
                    "learned_per_stage_post_centering",
                )
                self.assertEqual(manifest["gate_dc_offset_count"], 3)
                self.assertEqual(
                    manifest["gate_dc_offset_initialization"],
                    "zero",
                )
                self.assertEqual(
                    manifest["mask_mapping"],
                    "atan(pi*(centered+dc))/pi",
                )
                del parent, off, v2, v3
                gc.collect()

    def test_step_zero_outputs_match_relay_off_and_v2_exactly(self) -> None:
        parent = _small_parent(seed=73)
        off = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=False,
        )
        v2 = adapt_v8_mprs_dch_parent_v2(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        v3 = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        off.eval()
        v2.eval()
        v3.eval()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7301)
        inputs = torch.randn(2, 1, 32, 32, generator=generator)
        with torch.no_grad():
            off_outputs = off(inputs)
            v2_outputs = v2(inputs)
            v3_outputs = v3(inputs)
        self.assertEqual(len(off_outputs), 6)
        self.assertEqual(len(v2_outputs), 6)
        self.assertEqual(len(v3_outputs), 6)
        for index, (off_output, v2_output, v3_output) in enumerate(
            zip(off_outputs, v2_outputs, v3_outputs)
        ):
            self.assertTrue(
                torch.equal(off_output, v2_output),
                f"relay-off/V2 step-zero output {index}",
            )
            self.assertTrue(
                torch.equal(v2_output, v3_output),
                f"V2/V3 step-zero output {index}",
            )

        v3.train()
        v3.zero_grad(set_to_none=True)
        targets = torch.rand(
            2,
            1,
            32,
            32,
            generator=generator,
        )
        loss = _six_output_loss(v3(inputs), targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        total_offset_gradient = 0.0
        for stage, offset in v3.tpd_ner.dc_offsets.items():
            self.assertIsNotNone(offset.grad)
            assert offset.grad is not None
            self.assertTrue(torch.isfinite(offset.grad).all())
            stage_gradient = float(offset.grad.detach().abs().sum())
            self.assertGreater(
                stage_gradient,
                0.0,
                f"end-to-end DC offset gradient is zero at stage {stage}",
            )
            total_offset_gradient += stage_gradient
        self.assertGreater(total_offset_gradient, 0.0)

    def test_production_parameter_counts_and_strict_state_isolation(
        self,
    ) -> None:
        self.assertEqual(PRODUCTION_PARENT_PARAMETERS, TOTAL_PARAMETERS)
        self.assertEqual(
            PRODUCTION_V3_RELAY_PARAMETERS,
            PRODUCTION_V2_RELAY_PARAMETERS + 3,
        )
        parent, _ = build_clean_v8_mprs_dch_model(FULL, seed=42)
        off = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=False,
        )
        v2 = adapt_v8_mprs_dch_parent_v2(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        v3 = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        self.assertEqual(v3_relay_parameter_count(off), 0)
        self.assertEqual(
            v3_relay_parameter_count(v3),
            PRODUCTION_V3_RELAY_PARAMETERS,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in v3.parameters()),
            PRODUCTION_V3_RELAY_ON_PARAMETERS,
        )

        rebuilt = adapt_v8_mprs_dch_parent_v3(
            parent,
            variant=FULL,
            relay_enabled=True,
        )
        expected_offsets = {
            "4": -0.125,
            "3": 0.0625,
            "2": -0.03125,
        }
        with torch.no_grad():
            for stage, value in expected_offsets.items():
                v3.tpd_ner.dc_offsets[stage].fill_(value)
        incompatible = rebuilt.load_state_dict(
            v3.state_dict(),
            strict=True,
        )
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)
        for stage, expected in expected_offsets.items():
            self.assertEqual(
                float(rebuilt.tpd_ner.dc_offsets[stage].detach()),
                expected,
            )
        with self.assertRaises(RuntimeError):
            v2.load_state_dict(v3.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            v3.load_state_dict(v2.state_dict(), strict=True)
        del parent, off, v2, v3, rebuilt
        gc.collect()


if __name__ == "__main__":
    unittest.main()
