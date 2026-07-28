"""Explicit five-node NER adapter for frozen V8-MPRS-DCH parents.

The frozen V8 tokenizer already exposes five non-terminal evidence tensors:
``h11, h12, h13`` from ``embeddings_1`` and ``h21, h22`` from
``embeddings_2``.  This module copies a fully initialized V8 Full or Capacity
parent and optionally adds the existing narrow ``q4 -> q3 -> q2`` relay.

No frozen V8, V7, or SCTransNet source is modified.  Relay-on adds only
``tpd_ner.*`` state.  Its three spatial gates are zero initialized, so the
relay-on and relay-off variants have identical initial outputs and identical
shared-parameter optimization state after their first Adam step.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
    TPDCleanV8MPRSDCHPatchEmbedding,
    clean_v8_mprs_dch_variant_spec,
)
from model.tpd_sctransnet import (
    EVIDENCE_NODE_NAMES,
    RELAY_STAGE_ORDER,
    ExplicitNestedEvidenceRelay,
    ExplicitRelayUpBlock,
)


DEFAULT_RELAY_WIDTH = 8
# The complete-model initialization policy fixes this value to 42.  It is an
# explicit module-initialization field, while split_seed=20260722 belongs only
# to the external 530/133 data split and never enters model construction.
DEFAULT_RELAY_INITIALIZATION_SEED = 42
PRODUCTION_RELAY_PARAMETERS = 11_291
PRODUCTION_PARENT_PARAMETERS = 10_843_155
PRODUCTION_RELAY_ON_PARAMETERS = 10_854_446


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _validate_decoder_channels(
    block: nn.Module,
    name: str,
    expected_channels: int,
) -> None:
    try:
        g_linear = block.coatt.mlp_g[1]
        x_linear = block.coatt.mlp_x[1]
        first_conv = block.nConvs[0].conv
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError(f"{name} is not a compatible CCA decoder block") from exc
    if (
        not isinstance(g_linear, nn.Linear)
        or not isinstance(x_linear, nn.Linear)
        or g_linear.in_features != expected_channels
        or x_linear.in_features != expected_channels
        or first_conv.in_channels != 2 * expected_channels
    ):
        raise ValueError(
            f"{name} does not satisfy the expected C={expected_channels} contract"
        )


def _validate_v8_parent(
    parent: nn.Module,
    variant: str,
) -> Tuple[int, Mapping[str, object]]:
    if not isinstance(parent, SCTransNet):
        raise TypeError("V8 NER parent must be an SCTransNet")
    if hasattr(parent, "tpd_ner"):
        raise ValueError("V8 NER parent already registers tpd_ner")
    if parent._parameters or parent._buffers:
        raise TypeError(
            "V8 NER adapter requires SCTransNet parameters and buffers to be "
            "owned by registered child modules"
        )

    variant = variant.lower()
    spec = clean_v8_mprs_dch_variant_spec(variant)
    expected_context_gate = float(spec["context_gate"])
    channels = []
    for embedding_name, expected_blocks in (
        ("embeddings_1", 4),
        ("embeddings_2", 3),
    ):
        embedding = getattr(parent.mtc, embedding_name, None)
        if not isinstance(embedding, TPDCleanV8MPRSDCHPatchEmbedding):
            raise TypeError(f"{embedding_name} is not a V8-MPRS-DCH embedding")
        evidence_forward = getattr(embedding, "forward_with_evidence", None)
        if not callable(evidence_forward):
            raise TypeError(
                f"{embedding_name} does not expose forward_with_evidence"
            )
        if len(embedding.blocks) != expected_blocks:
            raise ValueError(
                f"{embedding_name} requires {expected_blocks} blocks, "
                f"got {len(embedding.blocks)}"
            )
        embedding_channels = None
        for index, block in enumerate(embedding.blocks):
            if not isinstance(block, TPDCleanV8MPRSDCHBlock):
                raise TypeError(
                    f"{embedding_name}.blocks[{index}] is not V8-MPRS-DCH"
                )
            if block.context_gate != expected_context_gate:
                raise ValueError(
                    f"{embedding_name}.blocks[{index}] context gate "
                    f"{block.context_gate} does not match {variant}"
                )
            if embedding_channels is None:
                embedding_channels = block.channels
            elif block.channels != embedding_channels:
                raise ValueError(
                    f"{embedding_name} changes channels between blocks"
                )
        if not isinstance(embedding_channels, int):
            raise RuntimeError(f"cannot infer channels from {embedding_name}")
        channels.append(embedding_channels)

    base_channels, second_channels = channels
    if second_channels != 2 * base_channels:
        raise ValueError(
            "embeddings_2 channels must equal twice embeddings_1 channels"
        )
    for block, name, expected_channels in (
        (parent.up_decoder4, "up_decoder4", 8 * base_channels),
        (parent.up_decoder3, "up_decoder3", 4 * base_channels),
        (parent.up_decoder2, "up_decoder2", 2 * base_channels),
    ):
        _validate_decoder_channels(block, name, expected_channels)
    return base_channels, spec


def _initialize_relay(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class TPDNERV8MPRSDCHSCTransNet(SCTransNet):
    """A copied V8 parent with an optional explicit five-node relay."""

    evidence_node_names = EVIDENCE_NODE_NAMES
    relay_stage_order = RELAY_STAGE_ORDER

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_enabled: bool,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
    ) -> None:
        if relay_width < 1:
            raise ValueError(f"relay_width must be positive, got {relay_width}")
        if relay_initialization_seed < 0:
            raise ValueError("relay_initialization_seed must be non-negative")
        variant = variant.lower()
        base_channels, spec = _validate_v8_parent(parent, variant)

        # Build an independent extension without reinitializing or changing
        # any parent tensor, including the seven learned saliency scales.
        nn.Module.__init__(self)
        for name, child in parent._modules.items():
            self.add_module(name, copy.deepcopy(child))
        for name in (
            "vis",
            "deepsuper",
            "mode",
            "n_channels",
            "n_classes",
        ):
            setattr(self, name, copy.deepcopy(getattr(parent, name)))
        self.training = bool(parent.training)

        self.tokenizer_variant = variant
        self.relay_enabled = bool(relay_enabled)
        self.relay_width = int(relay_width)
        self.relay_initialization_seed = int(relay_initialization_seed)
        self._nested_relay_installed = False
        self._variant_spec = dict(spec)

        if self.relay_enabled:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.relay_initialization_seed)
                relay = ExplicitNestedEvidenceRelay(
                    base_channels=base_channels,
                    width=self.relay_width,
                )
                relay.apply(_initialize_relay)
            reference = next(self.parameters())
            relay.to(device=reference.device, dtype=reference.dtype)
            relay.zero_init_gates()

            decoder4 = ExplicitRelayUpBlock.from_existing(
                self.up_decoder4,
                stage=4,
            )
            decoder3 = ExplicitRelayUpBlock.from_existing(
                self.up_decoder3,
                stage=3,
            )
            decoder2 = ExplicitRelayUpBlock.from_existing(
                self.up_decoder2,
                stage=2,
            )
            self.up_decoder4 = decoder4
            self.up_decoder3 = decoder3
            self.up_decoder2 = decoder2
            self.add_module("tpd_ner", relay)
            self._nested_relay_installed = True

        # Revalidate the copied tokenizer rather than trusting the source
        # object after construction.
        _validate_v8_parent_copy(self, variant)

    def zero_init_relay_gates(self) -> None:
        """Reset only NER gates; trained V8 saliency scales are untouched."""

        if not self.relay_enabled:
            if hasattr(self, "tpd_ner"):
                raise RuntimeError("relay-off unexpectedly registers tpd_ner")
            return
        if not self._nested_relay_installed:
            raise RuntimeError("relay is enabled but not installed")
        self.tpd_ner.zero_init_gates()

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "model": self.__class__.__name__,
            "tokenizer_variant": self.tokenizer_variant,
            "mainline_contract": self._variant_spec["mainline_contract"],
            "semantic_sources": self._variant_spec["semantic_sources"],
            "semantic_source_count": 3,
            "fourth_parallel_branch_added": False,
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "evidence_node_count": 5,
            "evidence_layout": (3, 2),
            "relay_enabled": self.relay_enabled,
            "relay_stage_order": RELAY_STAGE_ORDER,
            "relay_width": self.relay_width,
            "relay_state_prefix": "tpd_ner.",
            "tensor_handoff": "forward_local_explicit",
            "ordinary_forward_uses_mprs_diagnostics": False,
        }

    def explicit_embeddings(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        forward1 = self.mtc.embeddings_1.forward_with_evidence
        forward2 = self.mtc.embeddings_2.forward_with_evidence
        emb1, evidence1 = forward1(x1)
        emb2, evidence2 = forward2(x2)
        emb3 = self.mtc.embeddings_3(x3)
        emb4 = self.mtc.embeddings_4(x4)
        if emb1 is None or emb2 is None or emb3 is None or emb4 is None:
            raise RuntimeError("an embedding unexpectedly returned None")
        if not isinstance(evidence1, tuple) or not isinstance(evidence2, tuple):
            raise RuntimeError("evidence interface must return tuples")
        self._validate_evidence_shapes(
            endpoint=emb1,
            evidence=evidence1,
            source=x1,
            expected_evidence=3,
            name="embeddings_1",
        )
        self._validate_evidence_shapes(
            endpoint=emb2,
            evidence=evidence2,
            source=x2,
            expected_evidence=2,
            name="embeddings_2",
        )
        return emb1, emb2, emb3, emb4, evidence1, evidence2

    @staticmethod
    def _validate_evidence_shapes(
        *,
        endpoint: torch.Tensor,
        evidence: Sequence[torch.Tensor],
        source: torch.Tensor,
        expected_evidence: int,
        name: str,
    ) -> None:
        if source.ndim != 4:
            raise RuntimeError(f"{name} source is not BCHW")
        if len(evidence) != expected_evidence:
            raise RuntimeError(
                f"{name} requires {expected_evidence} evidence tensors"
            )
        batch, channels, height, width = source.shape
        expected_shapes = [
            (
                batch,
                channels,
                height // (2 ** (index + 1)),
                width // (2 ** (index + 1)),
            )
            for index in range(expected_evidence)
        ]
        actual_shapes = []
        for index, tensor in enumerate(evidence):
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"{name} evidence[{index}] is not a Tensor")
            if tensor.dtype != source.dtype or tensor.device != source.device:
                raise RuntimeError(
                    f"{name} evidence[{index}] dtype/device differs"
                )
            actual_shapes.append(tuple(tensor.shape))
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"{name} evidence shapes={actual_shapes}, "
                f"expected={expected_shapes}"
            )
        endpoint_shape = (
            batch,
            channels,
            height // (2 ** (expected_evidence + 1)),
            width // (2 ** (expected_evidence + 1)),
        )
        if tuple(endpoint.shape) != endpoint_shape:
            raise RuntimeError(
                f"{name} endpoint shape={tuple(endpoint.shape)}, "
                f"expected={endpoint_shape}"
            )
        if endpoint.dtype != source.dtype or endpoint.device != source.device:
            raise RuntimeError(f"{name} endpoint dtype/device differs")

    def _forward_with_relay(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))
        f1, f2, f3, f4 = x1, x2, x3, x4

        emb1, emb2, emb3, emb4, evidence1, evidence2 = (
            self.explicit_embeddings(x1, x2, x3, x4)
        )
        h11, h12, h13 = evidence1
        h21, h22 = evidence2
        encoded1, encoded2, encoded3, encoded4, _ = self.mtc.encoder(
            emb1,
            emb2,
            emb3,
            emb4,
        )
        x1 = self.mtc.reconstruct_1(encoded1) + f1
        x2 = self.mtc.reconstruct_2(encoded2) + f2
        x3 = self.mtc.reconstruct_3(encoded3) + f3
        x4 = self.mtc.reconstruct_4(encoded4) + f4
        x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

        up4, skip4 = self.up_decoder4.prepare(d5, x4)
        q4, mask4 = self.tpd_ner.forward_stage(
            4,
            (h13, h22, up4),
            tuple(up4.shape[-2:]),
        )
        d4 = self.up_decoder4.finish(up4, skip4, mask4)

        up3, skip3 = self.up_decoder3.prepare(d4, x3)
        q3, mask3 = self.tpd_ner.forward_stage(
            3,
            (h12, h21, q4, up3),
            tuple(up3.shape[-2:]),
        )
        d3 = self.up_decoder3.finish(up3, skip3, mask3)

        up2, skip2 = self.up_decoder2.prepare(d3, x2)
        _, mask2 = self.tpd_ner.forward_stage(
            2,
            (h11, q3, up2),
            tuple(up2.shape[-2:]),
        )
        d2 = self.up_decoder2.finish(up2, skip2, mask2)
        out = self.outc(self.up_decoder1(d2, x1))

        if not self.deepsuper:
            return torch.sigmoid(out)
        gt_5 = self.gt_conv5(d5)
        gt_4 = self.gt_conv4(d4)
        gt_3 = self.gt_conv3(d3)
        gt_2 = self.gt_conv2(d2)
        gt5 = F.interpolate(
            gt_5,
            scale_factor=16,
            mode="bilinear",
            align_corners=True,
        )
        gt4 = F.interpolate(
            gt_4,
            scale_factor=8,
            mode="bilinear",
            align_corners=True,
        )
        gt3 = F.interpolate(
            gt_3,
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        gt2 = F.interpolate(
            gt_2,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
        if self.mode != "train":
            return torch.sigmoid(out)
        return (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(out),
        )

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        if not self.relay_enabled:
            return SCTransNet.forward(self, x)
        if not self._nested_relay_installed:
            raise RuntimeError("V8-MPRS-DCH NER relay installation is incomplete")
        return self._forward_with_relay(x)


def _validate_v8_parent_copy(
    model: TPDNERV8MPRSDCHSCTransNet,
    variant: str,
) -> None:
    expected_gate = float(
        clean_v8_mprs_dch_variant_spec(variant)["context_gate"]
    )
    for embedding_name, expected_blocks in (
        ("embeddings_1", 4),
        ("embeddings_2", 3),
    ):
        embedding = getattr(model.mtc, embedding_name)
        if not isinstance(embedding, TPDCleanV8MPRSDCHPatchEmbedding):
            raise TypeError(f"copied {embedding_name} is not V8-MPRS-DCH")
        if len(embedding.blocks) != expected_blocks:
            raise ValueError(f"copied {embedding_name} block count differs")
        for block in embedding.blocks:
            if not isinstance(block, TPDCleanV8MPRSDCHBlock):
                raise TypeError(f"copied {embedding_name} contains foreign block")
            if block.context_gate != expected_gate:
                raise ValueError(f"copied {embedding_name} context gate differs")


def adapt_v8_mprs_dch_parent(
    parent: SCTransNet,
    *,
    variant: str,
    relay_enabled: bool,
    relay_width: int = DEFAULT_RELAY_WIDTH,
    relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
) -> TPDNERV8MPRSDCHSCTransNet:
    """Copy a verified V8 Full/Capacity parent into relay-off or relay-on."""

    return TPDNERV8MPRSDCHSCTransNet(
        parent,
        variant=variant,
        relay_enabled=relay_enabled,
        relay_width=relay_width,
        relay_initialization_seed=relay_initialization_seed,
    )


def relay_parameter_count(model: nn.Module) -> int:
    relay = getattr(model, "tpd_ner", None)
    return 0 if relay is None else _parameter_count(relay)


__all__ = [
    "DEFAULT_RELAY_INITIALIZATION_SEED",
    "DEFAULT_RELAY_WIDTH",
    "EVIDENCE_NODE_NAMES",
    "PRODUCTION_PARENT_PARAMETERS",
    "PRODUCTION_RELAY_ON_PARAMETERS",
    "PRODUCTION_RELAY_PARAMETERS",
    "RELAY_STAGE_ORDER",
    "SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS",
    "TPDNERV8MPRSDCHSCTransNet",
    "adapt_v8_mprs_dch_parent",
    "relay_parameter_count",
]
