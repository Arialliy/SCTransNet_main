"""Isolated V5/Progressive SCTransNet with an explicit five-node relay.

This module is deliberately additive.  It does not alter the frozen V5
tokenizer, the legacy NER implementation, or SCTransNet itself.

Both tokenizers expose the same five intermediate evidence tensors:

``h11, h12, h13`` from ``embeddings_1`` and ``h21, h22`` from
``embeddings_2``.  When enabled, the narrow relay consumes them in the
strictly acyclic order ``q4 -> q3 -> q2``.  Relay-off uses the ordinary
SCTransNet decoder.  Relay-on only wraps the same decoder modules and adds
``tpd_ner.*`` parameters, whose spatial gates initialize to zero.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v5 import (
    PRIMARY_CLEAN_V5_VARIANT,
    TPDCleanV5Block,
    build_clean_v5_patch_embedding,
)
from model.tpd_sctransnet import (
    EVIDENCE_NODE_NAMES,
    RELAY_STAGE_ORDER,
    ExplicitNestedEvidenceRelay,
    ExplicitRelayUpBlock,
)


PROGRESSIVE_TOKENIZER = "progressive_capacity_matched"
SUPPORTED_EVIDENCE_TOKENIZERS = (
    PRIMARY_CLEAN_V5_VARIANT,
    PROGRESSIVE_TOKENIZER,
)


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return stride.bit_length() - 1


class CapacityMatchedProgressiveBlock(nn.Module):
    """One learned stride-2 step with exactly the V5 block parameter count.

    For ``C`` channels the spatial projection contains ``4*C*C + C``
    parameters.  The bounded per-channel gain contributes another ``C`` and
    participates directly in every forward:

    ``Y = activation(Conv2x2,s=2(X) * (1 + tanh(gain)))``.

    Hence the total is ``4*C*C + 2*C``, exactly matching one V5 KCS block.
    The gain initializes to zero, so it begins as an identity multiplier
    while retaining a first-step gradient.
    """

    def __init__(self, channels: int, activate: bool) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.spatial_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
        )
        self.channel_gain = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "CapacityMatchedProgressiveBlock requires BCHW input, "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "CapacityMatchedProgressiveBlock requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )
        x = self.spatial_projection(x)
        gain = 1.0 + torch.tanh(self.channel_gain.float())
        x = x * gain.to(dtype=x.dtype).view(1, -1, 1, 1)
        return self.activation(x)


class CapacityMatchedProgressivePatchEmbedding(nn.Module):
    """Repeated capacity-matched stride-2 blocks at the V5 hierarchy depth."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            CapacityMatchedProgressiveBlock(
                channels,
                activate=index < steps - 1,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x


def _original_embedding_contract(
    embedding: nn.Module,
    name: str,
) -> Tuple[int, int]:
    projection = getattr(embedding, "patch_embeddings", None)
    channels = getattr(projection, "in_channels", None)
    stride = getattr(projection, "stride", None)
    if not isinstance(channels, int) or channels < 1:
        raise TypeError(f"cannot infer input channels from original {name}")
    if (
        not isinstance(stride, tuple)
        or len(stride) != 2
        or stride[0] != stride[1]
    ):
        raise TypeError(f"cannot infer square stride from original {name}")
    return channels, int(stride[0])


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


def _embedding_channels(embedding: nn.Module, name: str) -> int:
    blocks = getattr(embedding, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError(f"{name} must expose a non-empty ModuleList named 'blocks'")
    first = blocks[0]
    if isinstance(first, TPDCleanV5Block):
        channels = first.phase_compress.out_channels
    elif isinstance(first, CapacityMatchedProgressiveBlock):
        channels = first.spatial_projection.out_channels
    else:
        channels = None
    if not isinstance(channels, int) or channels < 1:
        raise TypeError(f"cannot infer channels from {name}.blocks[0]")
    return channels


def _validate_embedding(
    embedding: nn.Module,
    *,
    tokenizer_variant: str,
    name: str,
    expected_blocks: int,
) -> None:
    blocks = getattr(embedding, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(f"{name} must expose a ModuleList named 'blocks'")
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"{name} requires {expected_blocks} blocks, got {len(blocks)}"
        )
    if tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT:
        for index, block in enumerate(blocks):
            if not isinstance(block, TPDCleanV5Block):
                raise TypeError(f"{name}.blocks[{index}] is not a V5 KCS block")
            if block.use_context_selector is not True:
                raise TypeError(
                    f"{name}.blocks[{index}] must use the positive Context selector"
                )
            if not isinstance(block.saliency_scale, nn.Parameter):
                raise TypeError(
                    f"{name}.blocks[{index}] lacks the single saliency scale"
                )
            if hasattr(block, "context_scale"):
                raise TypeError(
                    f"{name}.blocks[{index}] unexpectedly adds a Context scale"
                )
    elif tokenizer_variant == PROGRESSIVE_TOKENIZER:
        for index, block in enumerate(blocks):
            if not isinstance(block, CapacityMatchedProgressiveBlock):
                raise TypeError(
                    f"{name}.blocks[{index}] is not capacity-matched Progressive"
                )
    else:
        raise ValueError(
            f"unknown evidence tokenizer {tokenizer_variant!r}; "
            f"choices={SUPPORTED_EVIDENCE_TOKENIZERS}"
        )


class ExplicitV5EvidenceEmbedding(nn.Module):
    """Hierarchical tokenizer that returns its non-terminal states explicitly."""

    def __init__(
        self,
        embedding: nn.Module,
        *,
        tokenizer_variant: str,
        name: str,
        expected_blocks: int,
    ) -> None:
        super().__init__()
        _validate_embedding(
            embedding,
            tokenizer_variant=tokenizer_variant,
            name=name,
            expected_blocks=expected_blocks,
        )
        # Moving the ModuleList preserves the stable
        # ``mtc.embeddings_{1,2}.blocks.*`` checkpoint keys.
        self.blocks = embedding.blocks
        self.tokenizer_variant = tokenizer_variant
        self.name = name

    def forward_with_evidence(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if x.ndim != 4:
            raise ValueError(f"{self.name} input must be BCHW, got {tuple(x.shape)}")
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states[-1], tuple(states[:-1])

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        endpoint, _ = self.forward_with_evidence(x)
        return endpoint


def _build_tokenizer_embedding(
    tokenizer_variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    if tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT:
        return build_clean_v5_patch_embedding(
            PRIMARY_CLEAN_V5_VARIANT,
            channels,
            stride,
        )
    if tokenizer_variant == PROGRESSIVE_TOKENIZER:
        return CapacityMatchedProgressivePatchEmbedding(channels, stride)
    raise ValueError(
        f"unknown evidence tokenizer {tokenizer_variant!r}; "
        f"choices={SUPPORTED_EVIDENCE_TOKENIZERS}"
    )


class TPDNERV5SCTransNet(SCTransNet):
    """SCTransNet with a V5/Progressive evidence tokenizer and optional relay."""

    evidence_node_names = EVIDENCE_NODE_NAMES
    relay_stage_order = RELAY_STAGE_ORDER

    def __init__(
        self,
        config: Any,
        n_channels: int = 1,
        n_classes: int = 1,
        img_size: int = 256,
        vis: bool = False,
        mode: str = "train",
        deepsuper: bool = True,
        *,
        tokenizer_variant: str = PRIMARY_CLEAN_V5_VARIANT,
        relay_enabled: bool = True,
        relay_width: int = 8,
        install_extension: bool = True,
    ) -> None:
        tokenizer_variant = tokenizer_variant.lower()
        if tokenizer_variant not in SUPPORTED_EVIDENCE_TOKENIZERS:
            raise ValueError(
                f"unknown evidence tokenizer {tokenizer_variant!r}; "
                f"choices={SUPPORTED_EVIDENCE_TOKENIZERS}"
            )
        if relay_width < 1:
            raise ValueError(f"relay_width must be positive, got {relay_width}")
        super().__init__(
            config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )
        self.tokenizer_variant = tokenizer_variant
        self.relay_enabled = bool(relay_enabled)
        self.relay_width = int(relay_width)
        self._evidence_tokenizer_installed = False
        self._nested_relay_installed = False
        if install_extension:
            self.install_evidence_tokenizer()
            if self.relay_enabled:
                self.install_nested_relay()

    def install_evidence_tokenizer(
        self,
    ) -> Dict[str, ExplicitV5EvidenceEmbedding]:
        if self._evidence_tokenizer_installed:
            raise ValueError("evidence tokenizer is already installed")
        if self._nested_relay_installed or hasattr(self, "tpd_ner"):
            raise RuntimeError("cannot replace tokenizer after relay installation")
        channels1, stride1 = _original_embedding_contract(
            self.mtc.embeddings_1,
            "embeddings_1",
        )
        channels2, stride2 = _original_embedding_contract(
            self.mtc.embeddings_2,
            "embeddings_2",
        )
        if channels2 != 2 * channels1:
            raise ValueError("embeddings_2 channels must equal 2 * embeddings_1")
        raw1 = _build_tokenizer_embedding(
            self.tokenizer_variant,
            channels1,
            stride1,
        )
        raw2 = _build_tokenizer_embedding(
            self.tokenizer_variant,
            channels2,
            stride2,
        )
        embedding1 = ExplicitV5EvidenceEmbedding(
            raw1,
            tokenizer_variant=self.tokenizer_variant,
            name="embeddings_1",
            expected_blocks=4,
        )
        embedding2 = ExplicitV5EvidenceEmbedding(
            raw2,
            tokenizer_variant=self.tokenizer_variant,
            name="embeddings_2",
            expected_blocks=3,
        )
        self.mtc.embeddings_1 = embedding1
        self.mtc.embeddings_2 = embedding2
        self._evidence_tokenizer_installed = True
        return {
            "embeddings_1": embedding1,
            "embeddings_2": embedding2,
        }

    def install_nested_relay(self) -> Dict[str, nn.Module]:
        if not self.relay_enabled:
            raise RuntimeError("relay is disabled for this paired variant")
        if not self._evidence_tokenizer_installed:
            raise RuntimeError("install the evidence tokenizer before the relay")
        if self._nested_relay_installed or hasattr(self, "tpd_ner"):
            raise ValueError("Nested Evidence Relay is already installed")
        embedding1 = self.mtc.embeddings_1
        embedding2 = self.mtc.embeddings_2
        if not isinstance(embedding1, ExplicitV5EvidenceEmbedding):
            raise TypeError("embeddings_1 is not an explicit evidence tokenizer")
        if not isinstance(embedding2, ExplicitV5EvidenceEmbedding):
            raise TypeError("embeddings_2 is not an explicit evidence tokenizer")
        base_channels = _embedding_channels(embedding1, "embeddings_1")
        second_channels = _embedding_channels(embedding2, "embeddings_2")
        if second_channels != 2 * base_channels:
            raise ValueError("invalid emb1/emb2 channel relationship")
        for block, name, channels in (
            (self.up_decoder4, "up_decoder4", 8 * base_channels),
            (self.up_decoder3, "up_decoder3", 4 * base_channels),
            (self.up_decoder2, "up_decoder2", 2 * base_channels),
        ):
            _validate_decoder_channels(block, name, channels)
        relay = ExplicitNestedEvidenceRelay(
            base_channels=base_channels,
            width=self.relay_width,
        )
        decoder4 = ExplicitRelayUpBlock.from_existing(self.up_decoder4, stage=4)
        decoder3 = ExplicitRelayUpBlock.from_existing(self.up_decoder3, stage=3)
        decoder2 = ExplicitRelayUpBlock.from_existing(self.up_decoder2, stage=2)
        self.add_module("tpd_ner", relay)
        self.up_decoder4 = decoder4
        self.up_decoder3 = decoder3
        self.up_decoder2 = decoder2
        self._nested_relay_installed = True
        relay.zero_init_gates()
        return {
            "embedding_1": embedding1,
            "embedding_2": embedding2,
            "relay": relay,
            "decoder_4": decoder4,
            "decoder_3": decoder3,
            "decoder_2": decoder2,
        }

    def zero_init_extension_residuals(self) -> None:
        if not self._evidence_tokenizer_installed:
            raise RuntimeError("evidence tokenizer is not installed")
        if self.tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT:
            for name in ("embeddings_1", "embeddings_2"):
                embedding = getattr(self.mtc, name)
                for block in embedding.blocks:
                    nn.init.zeros_(block.saliency_scale)
        elif self.tokenizer_variant == PROGRESSIVE_TOKENIZER:
            for name in ("embeddings_1", "embeddings_2"):
                embedding = getattr(self.mtc, name)
                for block in embedding.blocks:
                    nn.init.zeros_(block.channel_gain)
        if self.relay_enabled:
            if not self._nested_relay_installed:
                raise RuntimeError("relay is enabled but not installed")
            self.tpd_ner.zero_init_gates()

    def architecture_manifest(self) -> Dict[str, Any]:
        if not self._evidence_tokenizer_installed:
            raise RuntimeError("evidence tokenizer is not installed")
        if self.relay_enabled and not self._nested_relay_installed:
            raise RuntimeError("relay is enabled but not installed")
        return {
            "model": self.__class__.__name__,
            "tokenizer_variant": self.tokenizer_variant,
            "primary_module": (
                "Keep-Context-Saliency V5"
                if self.tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT
                else "Capacity-matched progressive convolution control"
            ),
            "secondary_module": (
                "Nested Evidence Relay" if self.relay_enabled else None
            ),
            "embedding_replacements": ("embeddings_1", "embeddings_2"),
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "relay_enabled": self.relay_enabled,
            "relay_stage_order": RELAY_STAGE_ORDER,
            "relay_width": self.relay_width,
            "tensor_handoff": "forward_local_explicit",
            "semantic_sources": (
                ("Keep", "Context", "Saliency")
                if self.tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT
                else ("CapacityMatchedProgressiveConv",)
            ),
            "fourth_parallel_branch_added": False,
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
        embedding1 = self.mtc.embeddings_1
        embedding2 = self.mtc.embeddings_2
        if not isinstance(embedding1, ExplicitV5EvidenceEmbedding):
            raise RuntimeError("explicit embeddings_1 is not installed")
        if not isinstance(embedding2, ExplicitV5EvidenceEmbedding):
            raise RuntimeError("explicit embeddings_2 is not installed")
        emb1, evidence1 = embedding1.forward_with_evidence(x1)
        emb2, evidence2 = embedding2.forward_with_evidence(x2)
        emb3 = self.mtc.embeddings_3(x3)
        emb4 = self.mtc.embeddings_4(x4)
        if emb3 is None or emb4 is None:
            raise RuntimeError("embeddings_3/4 unexpectedly returned None")
        if len(evidence1) != 3 or len(evidence2) != 2:
            raise RuntimeError("V5-NER requires exactly 3+2 evidence nodes")
        return emb1, emb2, emb3, emb4, evidence1, evidence2

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
        if not self._evidence_tokenizer_installed:
            raise RuntimeError("V5-NER evidence tokenizer installation incomplete")
        if not self.relay_enabled:
            return SCTransNet.forward(self, x)
        if not self._nested_relay_installed:
            raise RuntimeError("V5-NER relay installation incomplete")
        return self._forward_with_relay(x)


def relay_parameter_count(model: nn.Module) -> int:
    relay = getattr(model, "tpd_ner", None)
    if relay is None:
        return 0
    return sum(parameter.numel() for parameter in relay.parameters())


__all__ = [
    "EVIDENCE_NODE_NAMES",
    "RELAY_STAGE_ORDER",
    "PROGRESSIVE_TOKENIZER",
    "SUPPORTED_EVIDENCE_TOKENIZERS",
    "CapacityMatchedProgressiveBlock",
    "CapacityMatchedProgressivePatchEmbedding",
    "ExplicitV5EvidenceEmbedding",
    "TPDNERV5SCTransNet",
    "relay_parameter_count",
]
