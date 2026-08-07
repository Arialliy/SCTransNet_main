from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.irstd_core_ring_loss import (
    LOSS_WEIGHTS,
    compute_irstd_core_ring_loss,
    loss_manifest,
)
from experiments.irstd_error_atlas import build_irstd_error_atlas
from experiments.irstd_logit_counterfactual import corrupt_irstd_logits
from model.irstd_core_ring_repair import (
    IRSTDCoreRingRepairHead,
    PRODUCTION_PARAMETER_COUNT,
    PRODUCTION_PERSISTENT_BUFFER_COUNT,
    PRODUCTION_STATE_KEY_COUNT,
)


def _head_inputs(
    *,
    batch: int = 2,
    height: int = 9,
    width: int = 11,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)

    def random(channels: int) -> torch.Tensor:
        return torch.randn(
            batch,
            channels,
            height,
            width,
            generator=generator,
            dtype=torch.float32,
        )

    return {
        "image": random(1),
        "z_out": random(1),
        "z_d0": random(1),
        "z_gt2": random(1),
        "z_gt3": random(1),
        "z_gt4": random(1),
        "z_gt5": random(1),
        "local_feature": random(32),
    }


def test_repair_head_is_exact_identity_and_has_frozen_context_contract() -> None:
    head = IRSTDCoreRingRepairHead()
    inputs = _head_inputs()
    for value in inputs.values():
        value.requires_grad_(True)

    output = head.forward_with_diagnostics(**inputs)

    assert torch.equal(output.routed_logits, inputs["z_out"].detach())
    assert int(torch.count_nonzero(output.delta_logits)) == 0
    assert int(torch.count_nonzero(output.positive_delta)) == 0
    assert int(torch.count_nonzero(output.negative_delta)) == 0
    assert bool(torch.isfinite(output.core_gate).all())
    assert bool(torch.isfinite(output.halo_gate).all())
    assert sum(parameter.numel() for parameter in head.parameters()) == (
        PRODUCTION_PARAMETER_COUNT
    )
    assert len(head.state_dict()) == PRODUCTION_STATE_KEY_COUNT
    assert len(tuple(head.buffers())) == PRODUCTION_PERSISTENT_BUFFER_COUNT

    output.routed_logits.sum().backward()
    assert all(value.grad is None for value in inputs.values())
    for terminal in (
        head.positive_residual_head,
        head.negative_residual_head,
    ):
        assert terminal.weight.grad is not None
        assert bool(torch.isfinite(terminal.weight.grad).all())
        assert int(torch.count_nonzero(terminal.weight.grad)) > 0


def test_head_rejects_different_semantic_limits_even_non_strict() -> None:
    source = IRSTDCoreRingRepairHead()
    incompatible = IRSTDCoreRingRepairHead(positive_limit=1.0)
    with pytest.raises(RuntimeError, match="semantic checkpoint mismatch"):
        incompatible.load_state_dict(source.state_dict(), strict=False)
    assert float(incompatible.positive_limit) == pytest.approx(1.0)


def test_head_center_is_local_context_crop_equivalent_with_eight_pixel_halo() -> None:
    torch.manual_seed(42)
    head = IRSTDCoreRingRepairHead().eval()
    with torch.no_grad():
        head.positive_residual_head.weight.fill_(0.03)
        head.positive_residual_head.bias.fill_(0.01)
        head.negative_residual_head.weight.fill_(-0.02)
        head.negative_residual_head.bias.fill_(0.02)
    full = _head_inputs(batch=1, height=64, width=64)
    outer = {name: value[..., 8:56, 8:56] for name, value in full.items()}
    with torch.no_grad():
        full_output = head.forward_with_diagnostics(**full)
        outer_output = head.forward_with_diagnostics(**outer)
    for field in full_output.__dataclass_fields__:
        expected = getattr(full_output, field)[..., 16:48, 16:48]
        observed = getattr(outer_output, field)[..., 8:40, 8:40]
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=1.0e-6)


def test_repair_head_rejects_nonfinite_and_misaligned_inputs() -> None:
    head = IRSTDCoreRingRepairHead()
    inputs = _head_inputs(batch=1)
    inputs["image"][0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="image"):
        head.forward_with_diagnostics(**inputs)

    inputs = _head_inputs(batch=1)
    inputs["z_d0"] = inputs["z_d0"][..., :-1]
    with pytest.raises(ValueError, match="z_d0"):
        head.forward_with_diagnostics(**inputs)


def test_error_atlas_captures_attached_tail_and_keeps_baseline_optional() -> None:
    target = np.zeros((11, 15), dtype=np.bool_)
    target[5, 2:5] = True
    logits = np.full(target.shape, -4.0, dtype=np.float32)
    logits[5, 2:8] = 2.0  # matched target support plus one attached tail
    logits[1, 12:14] = 2.0  # separate unmatched false-positive component

    atlas = build_irstd_error_atlas(
        current_logits=logits,
        target_mask=target,
        ring_radius=1,
        far_background_radius=3,
    )

    expected_attached = np.zeros_like(target)
    expected_attached[5, 5:8] = True
    expected_detached = np.zeros_like(target)
    expected_detached[1, 12:14] = True
    assert np.array_equal(atlas.target_component_ids > 0, target)
    assert np.array_equal(atlas.attached_halo, expected_attached)
    assert np.array_equal(atlas.detached_false_positive, expected_detached)
    assert np.array_equal(
        atlas.halo_target,
        expected_attached | expected_detached,
    )
    assert not bool(np.any(atlas.halo_target & target))
    assert atlas.baseline_available is False
    assert atlas.baseline_rescue is None
    assert atlas.baseline_halo_advantage is None


def test_error_atlas_exposes_bound_baseline_maps_only_when_supplied() -> None:
    target = np.zeros((7, 7), dtype=np.bool_)
    target[3, 3] = True
    current = np.full(target.shape, -2.0, dtype=np.float32)
    current[0, 0] = 2.0
    baseline = np.full(target.shape, -2.0, dtype=np.float32)
    baseline[3, 3] = 2.0

    atlas = build_irstd_error_atlas(
        current_logits=current,
        target_mask=target,
        baseline_logits=baseline,
    )

    assert atlas.baseline_available is True
    assert atlas.baseline_rescue is not None
    assert atlas.baseline_halo_advantage is not None
    assert bool(atlas.baseline_rescue[3, 3])
    assert bool(atlas.baseline_halo_advantage[0, 0])
    assert bool(atlas.halo_target[0, 0])


def test_error_atlas_rejects_nonfinite_logits_and_nonbinary_target() -> None:
    logits = np.zeros((5, 5), dtype=np.float32)
    target = np.zeros((5, 5), dtype=np.float32)
    target[0, 0] = 0.25
    with pytest.raises(ValueError, match="binary"):
        build_irstd_error_atlas(
            current_logits=logits,
            target_mask=target,
        )

    target[0, 0] = 0.0
    logits[0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="current_logits"):
        build_irstd_error_atlas(
            current_logits=logits,
            target_mask=target,
        )


def _counterfactual_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core = torch.zeros(3, 1, 5, 5, dtype=torch.bool)
    ring = torch.zeros_like(core)
    observed = torch.zeros_like(core)
    core[:, :, 2, 2] = True
    ring[:, :, 1:4, 1:4] = True
    ring[:, :, 2, 2] = False
    observed[:, :, 0, 0] = True
    return core, ring, observed


def test_counterfactual_explicit_modes_are_balanced_local_and_reproducible() -> None:
    current = torch.zeros(3, 1, 5, 5, dtype=torch.float32)
    frozen_copy = current.clone()
    core, ring, observed = _counterfactual_masks()
    modes = torch.tensor([0, 1, 2], dtype=torch.int64)

    first = corrupt_irstd_logits(
        current_logits=current,
        core_target=core,
        outer_ring=ring,
        observed_halo_target=observed,
        generator=torch.Generator().manual_seed(23),
        modes=modes,
    )
    second = corrupt_irstd_logits(
        current_logits=current,
        core_target=core,
        outer_ring=ring,
        observed_halo_target=observed,
        generator=torch.Generator().manual_seed(23),
        modes=[0, 1, 2],
    )

    assert torch.equal(current, frozen_copy)
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.halo_target, second.halo_target)
    assert torch.equal(first.mode, modes)
    assert torch.equal(first.logits[0], current[0])

    core_difference = first.logits[1] - current[1]
    assert bool((core_difference[core[1]] <= -0.8).all())
    assert bool((core_difference[core[1]] >= -2.2).all())
    assert int(torch.count_nonzero(core_difference[~core[1]])) == 0

    ring_difference = first.logits[2] - current[2]
    assert bool((ring_difference[ring[2]] >= 0.5).all())
    assert bool((ring_difference[ring[2]] <= 1.5).all())
    assert int(torch.count_nonzero(ring_difference[~ring[2]])) == 0
    assert torch.equal(first.halo_target[0], observed[0])
    assert torch.equal(first.halo_target[1], observed[1])
    assert torch.equal(first.halo_target[2], observed[2] | ring[2])


def test_counterfactual_rejects_invalid_explicit_modes_and_overlapping_core() -> None:
    current = torch.zeros(3, 1, 5, 5)
    core, ring, observed = _counterfactual_masks()
    with pytest.raises(ValueError, match="0, 1, or 2"):
        corrupt_irstd_logits(
            current_logits=current,
            core_target=core,
            outer_ring=ring,
            observed_halo_target=observed,
            generator=torch.Generator().manual_seed(1),
            modes=[0, 3, 1],
        )

    ring[0, 0, 2, 2] = True
    with pytest.raises(ValueError, match="must not overlap"):
        corrupt_irstd_logits(
            current_logits=current,
            core_target=core,
            outer_ring=ring,
            observed_halo_target=observed,
            generator=torch.Generator().manual_seed(1),
            modes=[0, 1, 2],
        )


def _loss_inputs() -> dict[str, torch.Tensor]:
    shape = (1, 1, 7, 7)
    target = torch.zeros(shape, dtype=torch.bool)
    target[0, 0, 3, 3:5] = True
    component_ids = torch.zeros(shape, dtype=torch.int32)
    component_ids[target] = 1
    core = torch.zeros(shape, dtype=torch.bool)
    core[0, 0, 3, 3] = True
    attached = torch.zeros(shape, dtype=torch.bool)
    attached[0, 0, 3, 5] = True
    halo = attached.clone()
    halo[0, 0, 1, 1] = True
    far = torch.zeros(shape, dtype=torch.bool)
    far[0, 0, 0, 6] = True

    current = torch.full(shape, -2.0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        current[target] = 2.0
    routed = current.detach().clone()
    routed[target] -= 1.0
    routed[attached] = 1.0
    routed.requires_grad_(True)

    positive_delta = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
    negative_delta = torch.zeros(shape, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        positive_delta[attached] = 0.5
        positive_delta[far] = 0.25
        negative_delta[target] = 0.4
    delta = positive_delta - negative_delta
    return {
        "routed_logits": routed,
        "current_logits": current,
        "target": target,
        "target_component_ids": component_ids,
        "core_target": core,
        "halo_target": halo,
        "attached_halo": attached,
        "far_background": far,
        "core_gate_logits": torch.zeros(shape),
        "halo_gate_logits": torch.zeros(shape),
        "positive_delta": positive_delta,
        "negative_delta": negative_delta,
        "delta_logits": delta,
    }


def test_core_ring_loss_has_no_drop_attached_halo_and_cross_arm_protection() -> None:
    inputs = _loss_inputs()
    output = compute_irstd_core_ring_loss(**inputs)

    assert bool(torch.isfinite(output.total))
    assert float(output.target_peak_no_drop) > 0.0
    assert float(output.target_support_no_drop) > 0.0
    assert float(output.attached_halo_probability) > 0.0
    assert float(output.cross_arm_leak) > 0.0
    output.total.backward()
    assert inputs["current_logits"].grad is None
    assert inputs["routed_logits"].grad is not None
    assert bool(torch.isfinite(inputs["routed_logits"].grad).all())


def test_core_ring_loss_empty_component_and_region_paths_are_finite_zeros() -> None:
    shape = (2, 1, 5, 5)
    zero_float = torch.zeros(shape, dtype=torch.float32)
    zero_mask = torch.zeros(shape, dtype=torch.bool)
    output = compute_irstd_core_ring_loss(
        routed_logits=zero_float.clone().requires_grad_(True),
        current_logits=zero_float,
        target=zero_mask,
        target_component_ids=torch.zeros(shape, dtype=torch.int64),
        core_target=zero_mask,
        halo_target=zero_mask,
        attached_halo=zero_mask,
        far_background=torch.ones(shape, dtype=torch.bool),
        core_gate_logits=zero_float,
        halo_gate_logits=zero_float,
        positive_delta=zero_float,
        negative_delta=zero_float,
        delta_logits=zero_float,
    )

    assert bool(torch.isfinite(output.total))
    assert float(output.component_peak) == 0.0
    assert float(output.centroid) == 0.0
    assert float(output.target_peak_no_drop) == 0.0
    assert float(output.target_support_no_drop) == 0.0
    assert float(output.halo_probability) == 0.0
    assert float(output.attached_halo_probability) == 0.0
    assert float(output.direction) == 0.0
    assert float(output.cross_arm_leak) == 0.0


def test_core_ring_loss_manifest_has_fixed_weights_and_no_margin() -> None:
    manifest = loss_manifest()
    assert manifest["weights"] == dict(LOSS_WEIGHTS)
    assert manifest["performance_acceptance_margin"] is None
    assert manifest["baseline_maps_required"] is False


def test_core_ring_loss_rejects_conflicting_topology_masks() -> None:
    inputs = _loss_inputs()
    inputs["attached_halo"] = inputs["target"].clone()
    with pytest.raises(ValueError, match="attached_halo"):
        compute_irstd_core_ring_loss(**inputs)
