from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V8 = load_module(
    "tpd_clean_v8_mprs_dch",
    str(Path(__file__).with_name("tpd_clean_v8_mprs_dch.py")),
)


class V7DCHLayoutBlock(nn.Module):
    """Only the parameter/state layout needed for strict-compatibility tests."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.phase_compress = nn.Conv2d(4 * channels, channels, 1)
        self.saliency_scale = nn.Parameter(torch.zeros(channels))


def close(a, b, *, atol=1e-6, rtol=1e-6, name="tensor"):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name}: max_abs={(a - b).abs().max().item():.9g}"
        )


torch.manual_seed(20260727)

# State layout / strict load.
legacy = V7DCHLayoutBlock(5)
block = V8.TPDCleanV8MPRSDCHBlock(
    5, activate=False, context_gate=1.0
)
assert list(legacy.state_dict()) == list(block.state_dict())
block.load_state_dict(legacy.state_dict(), strict=True)

# Mass, non-negativity, and flat-cell invariants.
x = torch.randn(3, 5, 18, 22)
_, _, s0, sp = block.phase_sources(x)
close(sp.sum(dim=2), 4.0 * s0.float(), atol=2e-6, rtol=2e-6, name="mass")
assert sp.min().item() >= -2e-6

base = torch.randn(2, 5, 7, 9)
equal_phases = base.unsqueeze(2).expand(-1, -1, 4, -1, -1).contiguous()
flat_x = F.pixel_shuffle(equal_phases.reshape(2, 20, 7, 9), 2)
_, _, flat_s0, flat_sp = block.phase_sources(flat_x)
close(flat_s0, torch.zeros_like(flat_s0), atol=2e-6, rtol=0, name="flat_s0")
close(flat_sp, torch.zeros_like(flat_sp), atol=2e-6, rtol=0, name="flat_sp")

# Phase permutation equivariance.
x = torch.randn(2, 5, 16, 20)
phases = F.pixel_unshuffle(x, 2).reshape(2, 5, 4, 8, 10)
perm = torch.tensor([2, 0, 3, 1])
x_perm = F.pixel_shuffle(phases[:, :, perm].reshape(2, 20, 8, 10), 2)
_, _, s0_a, sp_a = block.phase_sources(x)
_, _, s0_b, sp_b = block.phase_sources(x_perm)
close(s0_a, s0_b, name="perm_s0")
close(sp_b, sp_a[:, :, perm], name="perm_sp")

# Optimized algebraic path matches the explicit four-phase reference.
reference_block = V8.TPDCleanV8MPRSDCHBlock(
    5, activate=False, context_gate=1.0
)
reference_x = torch.randn(2, 5, 16, 20)
_, _, _, explicit_sp = reference_block.phase_sources(reference_x)
direct_aligned = F.conv2d(
    explicit_sp.reshape(2, 20, 8, 10),
    reference_block.phase_compress.weight.float(),
    bias=None,
)
(
    reference_keep,
    _,
    _,
    reference_correction,
    optimized_aligned,
) = reference_block.aligned_mprs_terms(reference_x)
close(
    optimized_aligned,
    direct_aligned,
    atol=5e-5,
    rtol=1e-5,
    name="optimized_vs_direct",
)
assert reference_correction.shape == optimized_aligned.shape

# Keep-bias reuse does not make Saliency depend on the Keep bias.
with torch.no_grad():
    aligned_before = reference_block.aligned_mprs_terms(reference_x)[-1]
    reference_block.phase_compress.bias.add_(torch.randn_like(
        reference_block.phase_compress.bias
    ))
    aligned_after = reference_block.aligned_mprs_terms(reference_x)[-1]
close(
    aligned_before,
    aligned_after,
    atol=5e-5,
    rtol=1e-5,
    name="bias_invariance",
)
reference_block.zero_grad(set_to_none=True)
reference_block.aligned_mprs_terms(reference_x)[-1].mean().backward()
assert reference_block.phase_compress.bias.grad is not None
assert (
    reference_block.phase_compress.bias.grad.abs().max().item()
    <= 1e-6
)

# Optimized and explicit-reference input/weight gradients agree.
optimized_block = V8.TPDCleanV8MPRSDCHBlock(
    5, activate=False, context_gate=1.0
)
direct_block = V8.TPDCleanV8MPRSDCHBlock(
    5, activate=False, context_gate=1.0
)
direct_block.load_state_dict(optimized_block.state_dict(), strict=True)
optimized_x = torch.randn(2, 5, 16, 20, requires_grad=True)
direct_x = optimized_x.detach().clone().requires_grad_(True)
optimized_sa = optimized_block.aligned_mprs_terms(optimized_x)[-1]
_, _, _, direct_sp = direct_block.phase_sources(direct_x)
direct_sa = F.conv2d(
    direct_sp.reshape(2, 20, 8, 10),
    direct_block.phase_compress.weight.float(),
    bias=None,
)
optimized_loss = optimized_sa.square().mean()
direct_loss = direct_sa.square().mean()
optimized_grad_x, optimized_grad_w = torch.autograd.grad(
    optimized_loss,
    (optimized_x, optimized_block.phase_compress.weight),
)
direct_grad_x, direct_grad_w = torch.autograd.grad(
    direct_loss,
    (direct_x, direct_block.phase_compress.weight),
)
close(
    optimized_grad_x,
    direct_grad_x,
    atol=2e-5,
    rtol=1e-4,
    name="optimized_direct_input_grad",
)
close(
    optimized_grad_w,
    direct_grad_w,
    atol=2e-5,
    rtol=1e-4,
    name="optimized_direct_weight_grad",
)

# Equal phase weights reduce exactly to V7-DCH alignment.
with torch.no_grad():
    tied = torch.randn(5, 5, 1, 1)
    block.phase_compress.weight.copy_(
        tied.unsqueeze(2).expand(-1, -1, 4, -1, -1).reshape(5, 20, 1, 1)
    )
    block.phase_compress.bias.zero_()
_, _, old_sa, new_sa = block.aligned_saliency_terms(
    torch.randn(2, 5, 16, 20)
)
close(old_sa, new_sa, atol=3e-6, rtol=3e-6, name="equal_weight")

# Zero-scale dense-SPD identity.
full = V8.TPDCleanV8MPRSDCHBlock(5, activate=True, context_gate=1.0)
capacity = V8.TPDCleanV8MPRSDCHBlock(5, activate=True, context_gate=0.0)
capacity.load_state_dict(full.state_dict(), strict=True)
x = torch.randn(2, 5, 16, 20)
spd = full.activation(full.phase_compress(F.pixel_unshuffle(x, 2)))
assert torch.equal(full(x), spd)
assert torch.equal(capacity(x), spd)

# Zero-scale gradients.
full = V8.TPDCleanV8MPRSDCHBlock(5, activate=False, context_gate=1.0)
capacity = V8.TPDCleanV8MPRSDCHBlock(5, activate=False, context_gate=0.0)
capacity.load_state_dict(full.state_dict(), strict=True)
xf = torch.randn(2, 5, 16, 20, requires_grad=True)
xc = xf.detach().clone().requires_grad_(True)
target = torch.randn(2, 5, 8, 10)
((full(xf) - target) ** 2).mean().backward()
((capacity(xc) - target) ** 2).mean().backward()
close(xf.grad, xc.grad, atol=1e-7, rtol=1e-6, name="input_grad")
for (nf, pf), (nc, pc) in zip(
    full.named_parameters(), capacity.named_parameters()
):
    assert nf == nc and pf.grad is not None and pc.grad is not None
    close(pf.grad, pc.grad, atol=1e-7, rtol=1e-6, name=f"grad:{nf}")

# First Adam step.
full = V8.TPDCleanV8MPRSDCHBlock(5, activate=False, context_gate=1.0)
capacity = V8.TPDCleanV8MPRSDCHBlock(5, activate=False, context_gate=0.0)
capacity.load_state_dict(full.state_dict(), strict=True)
opt_f = torch.optim.Adam(full.parameters(), lr=1e-3)
opt_c = torch.optim.Adam(capacity.parameters(), lr=1e-3)
x = torch.randn(2, 5, 16, 20)
target = torch.randn(2, 5, 8, 10)
for module, optimizer in ((full, opt_f), (capacity, opt_c)):
    optimizer.zero_grad(set_to_none=True)
    ((module(x) - target) ** 2).mean().backward()
    optimizer.step()
for (nf, pf), (nc, pc) in zip(
    full.named_parameters(), capacity.named_parameters()
):
    assert nf == nc
    close(pf, pc, atol=1e-8, rtol=1e-7, name=f"adam:{nf}")

# Nonzero-scale Full/Capacity divergence.
with torch.no_grad():
    full.saliency_scale.fill_(0.6)
    capacity.saliency_scale.fill_(0.6)
assert not torch.equal(full(x), capacity(x))

# Standard forward uses exactly three convolutions per block and never calls
# the explicit diagnostic phase-source path.
for gate in (0.0, 1.0):
    counted = V8.TPDCleanV8MPRSDCHBlock(
        5,
        activate=False,
        context_gate=gate,
    )
    counted_input = torch.randn(2, 5, 16, 20)
    with (
        mock.patch.object(
            counted,
            "phase_sources",
            side_effect=AssertionError(
                "standard forward used explicit phase sources"
            ),
        ),
        mock.patch.object(
            counted,
            "phase_tied_weight",
            wraps=counted.phase_tied_weight,
        ) as tied_calls,
        mock.patch.object(
            V8.F,
            "conv2d",
            wraps=V8.F.conv2d,
        ) as conv_calls,
    ):
        counted(counted_input)
    assert tied_calls.call_count == 1
    assert conv_calls.call_count == 3

# The diagnostics interface reuses one computation and matches forward.
diagnostic_block = V8.TPDCleanV8MPRSDCHBlock(
    5,
    activate=False,
    context_gate=1.0,
)
diagnostic_x = torch.randn(2, 5, 16, 20)
diagnostic_output, diagnostics = (
    diagnostic_block.forward_with_mprs_diagnostics(diagnostic_x)
)
close(
    diagnostic_output,
    diagnostic_block(diagnostic_x),
    name="diagnostic_output",
)
assert set(diagnostics) == {
    "context_aligned",
    "saliency_v7",
    "phase_correction",
    "saliency_v8",
    "scale",
    "modulation",
    "headroom",
}

# Embedding shape, evidence, and parameter count.
emb1 = V8.build_clean_v8_mprs_dch_patch_embedding(
    "tpd_clean_v8_mprs_dch_full", 32, 16
)
emb2 = V8.build_clean_v8_mprs_dch_patch_embedding(
    "tpd_clean_v8_mprs_dch_full", 64, 8
)
y1, e1 = emb1.forward_with_evidence(torch.randn(1, 32, 256, 256))
y2, e2 = emb2.forward_with_evidence(torch.randn(1, 64, 128, 128))
assert y1.shape == (1, 32, 16, 16)
assert y2.shape == (1, 64, 16, 16)
assert len(e1) == 3 and len(e2) == 2
assert V8.parameter_count(emb1) + V8.parameter_count(emb2) == 66_176

print("PASS: state compatibility")
print("PASS: saliency mass/non-negativity/flat invariants")
print("PASS: phase-permutation equivariance")
print("PASS: optimized/direct forward and gradient equivalence")
print("PASS: bias cancellation and three-convolution production path")
print("PASS: equal-weight reduction to V7-DCH")
print("PASS: zero-scale dense-SPD identity")
print("PASS: zero-scale Full/Capacity gradients")
print("PASS: first Adam-step equality")
print("PASS: nonzero-scale Full/Capacity divergence")
print("PASS: single-pass MPRS diagnostics")
print("PASS: embedding shapes/evidence/parameter count")
