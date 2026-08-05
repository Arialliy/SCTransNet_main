from __future__ import annotations

import unittest
from typing import Sequence

import torch
import torch.nn as nn

from model.tpd_ner_v8_mprs_dch_v2 import (
    arctangent_residual_mask,
    spatially_center_gate_logits,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    TailAwarePersistentDCOffsetEvidenceRelay,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (
    PersistentEvidencePositiveRoutingRelay,
    replace_v4_relay_with_v5,
    v5_per_manifest_fields,
)


torch.set_num_threads(1)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _sources(
    stage: int,
    *,
    batch: int = 2,
    size: tuple[int, int] = (5, 7),
) -> tuple[torch.Tensor, ...]:
    channels = {
        4: (2, 4, 16),
        3: (2, 4, 8, 8),
        2: (2, 8, 4),
    }[stage]
    return tuple(torch.randn(batch, value, *size) for value in channels)


class _FixedFusion(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del sources
        if tuple(self.value.shape[-2:]) != tuple(output_size):
            raise ValueError("fixed relay value has the wrong spatial size")
        return self.value


class _ControlledSupportRelay(PersistentEvidencePositiveRoutingRelay):
    def __init__(self, support: torch.Tensor, centered: torch.Tensor) -> None:
        super().__init__(base_channels=2)
        relay_value = torch.zeros(
            centered.shape[0],
            8,
            centered.shape[2],
            centered.shape[3],
        )
        relay_value[:, :1] = centered
        self.fusions["2"] = _FixedFusion(relay_value)
        self._controlled_support = support
        with torch.no_grad():
            self.gates["2"].weight.zero_()
            self.gates["2"].weight[:, :1].fill_(1.0)

    def dc_support(
        self,
        stage: int,
        relay_value: torch.Tensor,
        sources: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        del relay_value, sources, output_size
        if stage != 2:
            raise AssertionError("controlled support is stage-2 only")
        return self._controlled_support


class NERV5PERModelTests(unittest.TestCase):
    def test_complement_only_and_semantic_manifest(self) -> None:
        for mode in ("legacy_global", "direct_tail"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "complement_tail"):
                    PersistentEvidencePositiveRoutingRelay(
                        base_channels=2,
                        dc_support_mode=mode,
                    )
        fields = v5_per_manifest_fields()
        self.assertEqual(fields["parameters_added_vs_v4"], 0)
        self.assertEqual(fields["buffers_added_vs_v4"], 0)
        self.assertFalse(fields["state_semantics_identical_to_v4"])
        self.assertFalse(
            fields["checkpoint_semantically_interchangeable_with_v4"]
        )
        self.assertFalse(fields["v4_to_v5_optimizer_resume_allowed"])

    def test_replacement_is_rng_neutral_and_state_layout_exact(self) -> None:
        torch.manual_seed(4201)
        v4 = TailAwarePersistentDCOffsetEvidenceRelay(base_channels=2)
        with torch.no_grad():
            v4.gates["2"].weight.normal_()
            v4.dc_offsets["2"].fill_(0.125)
        v4.eval()
        before_rng = torch.get_rng_state().clone()
        v5 = replace_v4_relay_with_v5(v4)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(tuple(v4.state_dict()), tuple(v5.state_dict()))
        self.assertEqual(_parameter_count(v4), _parameter_count(v5))
        self.assertFalse(tuple(v4.named_buffers()))
        self.assertFalse(tuple(v5.named_buffers()))
        self.assertFalse(v5.training)
        for name, value in v4.state_dict().items():
            self.assertTrue(torch.equal(value, v5.state_dict()[name]), name)

    def test_stage4_and_stage3_are_bitwise_v4(self) -> None:
        torch.manual_seed(9127)
        v4 = TailAwarePersistentDCOffsetEvidenceRelay(base_channels=2)
        with torch.no_grad():
            for stage in ("4", "3"):
                v4.gates[stage].weight.normal_(std=0.2)
                v4.dc_offsets[stage].fill_(0.075)
        v5 = replace_v4_relay_with_v5(v4)
        sources4 = _sources(4)
        q4_v4, mask4_v4 = v4.forward_stage(4, sources4, (5, 7))
        q4_v5, mask4_v5 = v5.forward_stage(4, sources4, (5, 7))
        self.assertTrue(torch.equal(q4_v4, q4_v5))
        self.assertTrue(torch.equal(mask4_v4, mask4_v5))

        sources3 = list(_sources(3))
        sources3[2] = q4_v4
        q3_v4, mask3_v4 = v4.forward_stage(3, sources3, (5, 7))
        q3_v5, mask3_v5 = v5.forward_stage(3, sources3, (5, 7))
        self.assertTrue(torch.equal(q3_v4, q3_v5))
        self.assertTrue(torch.equal(mask3_v4, mask3_v5))

    def test_stage2_formula_at_both_algebraic_support_limits(self) -> None:
        centered = torch.tensor(
            [[[[2.0, -1.0], [-3.0, 2.0]]]],
            dtype=torch.float32,
        )
        for background, dc in ((0.0, 0.7), (1.0, 0.25)):
            with self.subTest(background=background):
                support = torch.full_like(centered, background)
                relay = _ControlledSupportRelay(support, centered)
                with torch.no_grad():
                    relay.dc_offsets["2"].fill_(dc)
                _, actual = relay.forward_stage(2, (), (2, 2))
                c = spatially_center_gate_logits(centered)
                shifted = c - support * torch.relu(c) + dc * support
                expected = arctangent_residual_mask(shifted)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                if background == 0.0:
                    torch.testing.assert_close(
                        shifted,
                        c,
                        rtol=0,
                        atol=0,
                    )
                else:
                    routed = shifted - dc
                    self.assertTrue(torch.equal(routed[c > 0], torch.zeros_like(routed[c > 0])))
                    self.assertTrue(torch.equal(routed[c < 0], c[c < 0]))

    def test_real_support_range_zero_anchor_and_finite_gradients(self) -> None:
        torch.manual_seed(7123)
        relay = PersistentEvidencePositiveRoutingRelay(base_channels=2)
        sources = _sources(2, batch=2, size=(4, 4))
        relay_value = relay.fusions["2"](sources, (4, 4))
        background = relay.dc_support(2, relay_value, sources, (4, 4))
        persistent = 1.0 - background
        self.assertFalse(background.requires_grad)
        self.assertTrue(bool((persistent >= 0.0).all()))
        self.assertTrue(bool((persistent < 1.0).all()))
        self.assertTrue(bool((background > 0.0).all()))
        self.assertTrue(bool((background <= 1.0).all()))

        _, zero_mask = relay.forward_stage(2, sources, (4, 4))
        self.assertEqual(int(torch.count_nonzero(zero_mask)), 0)

        with torch.no_grad():
            relay.gates["2"].weight.normal_(std=0.1)
            relay.dc_offsets["2"].fill_(0.05)
        _, mask = relay.forward_stage(2, sources, (4, 4))
        weights = torch.linspace(0.1, 1.7, mask.numel()).reshape_as(mask)
        (mask * weights).sum().backward()
        for parameter in (
            relay.gates["2"].weight,
            relay.dc_offsets["2"],
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
