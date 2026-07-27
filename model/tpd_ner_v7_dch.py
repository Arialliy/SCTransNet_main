"""Isolated V7-DCH SCTransNet composer with an explicit five-node NER.

This module is intentionally additive.  It does not modify the frozen
TPD-Clean V7-DCH implementation or its active formal800 execution path.
The V7-DCH tokenizer remains a three-source Keep--Context--Saliency module.
Its two hierarchical embeddings expose three plus two non-terminal states:

``h11, h12, h13`` and ``h21, h22``.

When enabled, Nested Evidence Relay (NER) consumes those states in the
strictly acyclic order ``q4 -> q3 -> q2`` and modulates the existing decoder
skips after CCA.  Relay-off contains no ``tpd_ner.*`` parameters and executes
the ordinary SCTransNet decoder.  Relay-on adds only ``tpd_ner.*`` parameters;
its spatial gates initialize to zero, so paired relay-off/on models have
identical step-zero outputs.

Formal NER training is deliberately not launched from this module.  It is a
pre-gate implementation artifact and remains isolated until V7-DCH Gates A--E
authorize the NER stage.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v7_dch import (
    PRIMARY_CLEAN_V7_DCH_VARIANT,
    build_clean_v7_dch_patch_embedding,
)
from model.tpd_sctransnet import (
    EVIDENCE_NODE_NAMES,
    RELAY_STAGE_ORDER,
    ExplicitNestedEvidenceRelay,
    ExplicitRelayUpBlock,
)


PROGRESSIVE_TOKENIZER = "progressive_capacity_matched"
SUPPORTED_EVIDENCE_TOKENIZERS = (
    PRIMARY_CLEAN_V7_DCH_VARIANT,
    PROGRESSIVE_TOKENIZER,
)


class EvidencePatchEmbeddingProtocol(Protocol):
    """Capability contract consumed by the V7-DCH NER composer."""

    blocks: nn.ModuleList

    def forward(
        self,
        x: torch.Tensor | None,
    ) -> torch.Tensor | None:
        ...

    def forward_with_evidence(
        self,
        x: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, Tuple[torch.Tensor, ...]]:
        ...


def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return stride.bit_length() - 1


class CapacityMatchedProgressiveBlock(nn.Module):
    """One learned stride-2 control block matched to one V7-DCH block."""

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
        projected = self.spatial_projection(x)
        gain = 1.0 + torch.tanh(self.channel_gain.float())
        projected = projected * gain.to(dtype=projected.dtype).view(
            1,
            -1,
            1,
            1,
        )
        return self.activation(projected)


class CapacityMatchedProgressivePatchEmbedding(nn.Module):
    """Repeated capacity-matched stride-2 control with explicit evidence."""

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

    def forward(
        self,
        x: torch.Tensor | None,
    ) -> torch.Tensor | None:
        endpoint, _ = self.forward_with_evidence(x)
        return endpoint

    def forward_with_evidence(
        self,
        x: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, Tuple[torch.Tensor, ...]]:
        if x is None:
            return None, ()
        states = []
        for block in self.blocks:
            x = block(x)
            states.append(x)
        return states[-1], tuple(states[:-1])


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


def _block_channels(block: nn.Module, name: str) -> int:
    phase_projection = getattr(block, "phase_compress", None)
    progressive_projection = getattr(block, "spatial_projection", None)
    candidates = (
        getattr(phase_projection, "out_channels", None),
        getattr(progressive_projection, "out_channels", None),
        getattr(block, "channels", None),
    )
    channels = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, int) and candidate > 0
        ),
        None,
    )
    if channels is None:
        raise TypeError(f"cannot infer channels from {name}")
    return channels


def _embedding_channels(embedding: nn.Module, name: str) -> int:
    blocks = getattr(embedding, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError(f"{name} must expose a non-empty ModuleList named 'blocks'")
    return _block_channels(blocks[0], f"{name}.blocks[0]")


def _validate_evidence_embedding(
    embedding: nn.Module,
    *,
    tokenizer_variant: str,
    name: str,
    expected_blocks: int,
) -> int:
    channels = _validate_evidence_capability(
        embedding,
        name=name,
        expected_blocks=expected_blocks,
    )
    blocks = embedding.blocks

    for index, block in enumerate(blocks):
        block_name = f"{name}.blocks[{index}]"
        if tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT:
            phase_projection = getattr(block, "phase_compress", None)
            saliency_scale = getattr(block, "saliency_scale", None)
            if not isinstance(phase_projection, nn.Conv2d):
                raise TypeError(f"{block_name} lacks phase_compress Conv2d")
            if phase_projection.in_channels != 4 * channels:
                raise ValueError(f"{block_name} phase input is not four phases")
            if phase_projection.out_channels != channels:
                raise ValueError(f"{block_name} phase output differs")
            if tuple(phase_projection.kernel_size) != (1, 1):
                raise ValueError(f"{block_name} phase projection is not 1x1")
            if not isinstance(saliency_scale, nn.Parameter):
                raise TypeError(f"{block_name} lacks saliency_scale")
            if tuple(saliency_scale.shape) != (channels,):
                raise ValueError(f"{block_name} saliency_scale shape differs")
            if getattr(block, "context_gate", None) != 1.0:
                raise ValueError(f"{block_name} is not V7-DCH Full")
            if hasattr(block, "context_scale"):
                raise ValueError(f"{block_name} adds a fourth learned control")
        elif tokenizer_variant == PROGRESSIVE_TOKENIZER:
            spatial_projection = getattr(block, "spatial_projection", None)
            channel_gain = getattr(block, "channel_gain", None)
            if not isinstance(spatial_projection, nn.Conv2d):
                raise TypeError(f"{block_name} lacks spatial_projection Conv2d")
            if tuple(spatial_projection.kernel_size) != (2, 2):
                raise ValueError(f"{block_name} projection is not 2x2")
            if tuple(spatial_projection.stride) != (2, 2):
                raise ValueError(f"{block_name} projection stride is not 2")
            if not isinstance(channel_gain, nn.Parameter):
                raise TypeError(f"{block_name} lacks channel_gain")
            if tuple(channel_gain.shape) != (channels,):
                raise ValueError(f"{block_name} channel_gain shape differs")
        else:
            raise ValueError(
                f"unknown evidence tokenizer {tokenizer_variant!r}; "
                f"choices={SUPPORTED_EVIDENCE_TOKENIZERS}"
            )
    return channels


def _validate_evidence_capability(
    embedding: nn.Module,
    *,
    name: str,
    expected_blocks: int,
) -> int:
    """Validate only the interface that NER consumes, not tokenizer identity."""

    blocks = getattr(embedding, "blocks", None)
    evidence_forward = getattr(embedding, "forward_with_evidence", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(f"{name} must expose a ModuleList named 'blocks'")
    if not callable(evidence_forward):
        raise TypeError(f"{name} must expose callable forward_with_evidence")
    if len(blocks) != expected_blocks:
        raise ValueError(
            f"{name} requires {expected_blocks} blocks, got {len(blocks)}"
        )
    channels = _embedding_channels(embedding, name)

    for index, block in enumerate(blocks):
        block_name = f"{name}.blocks[{index}]"
        if _block_channels(block, block_name) != channels:
            raise ValueError(f"{block_name} changes the embedding channel count")
    return channels


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


def _build_evidence_embedding(
    tokenizer_variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    if tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT:
        return build_clean_v7_dch_patch_embedding(
            PRIMARY_CLEAN_V7_DCH_VARIANT,
            channels,
            stride,
        )
    if tokenizer_variant == PROGRESSIVE_TOKENIZER:
        return CapacityMatchedProgressivePatchEmbedding(channels, stride)
    raise ValueError(
        f"unknown evidence tokenizer {tokenizer_variant!r}; "
        f"choices={SUPPORTED_EVIDENCE_TOKENIZERS}"
    )


class TPDNERV7DCHSCTransNet(SCTransNet):
    """SCTransNet with V7-DCH/Progressive tokenization and optional NER."""

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
        tokenizer_variant: str = PRIMARY_CLEAN_V7_DCH_VARIANT,
        relay_enabled: bool = True,
        relay_width: int = 8,
        install_extension: bool = False,
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

    def install_evidence_tokenizer(self) -> Dict[str, nn.Module]:
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
        embedding1 = _build_evidence_embedding(
            self.tokenizer_variant,
            channels1,
            stride1,
        )
        embedding2 = _build_evidence_embedding(
            self.tokenizer_variant,
            channels2,
            stride2,
        )
        _validate_evidence_embedding(
            embedding1,
            tokenizer_variant=self.tokenizer_variant,
            name="embeddings_1",
            expected_blocks=4,
        )
        _validate_evidence_embedding(
            embedding2,
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
        base_channels = _validate_evidence_embedding(
            embedding1,
            tokenizer_variant=self.tokenizer_variant,
            name="embeddings_1",
            expected_blocks=4,
        )
        second_channels = _validate_evidence_embedding(
            embedding2,
            tokenizer_variant=self.tokenizer_variant,
            name="embeddings_2",
            expected_blocks=3,
        )
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

    def zero_init_fresh_tokenizer_controls(self) -> None:
        """Reset tokenizer controls only during fresh paired construction.

        This method must not be called after loading a trained V7-DCH parent,
        because doing so would erase its learned ``saliency_scale`` tensors.
        Parent-to-NER warm start is responsible only for zeroing relay gates.
        """

        if not self._evidence_tokenizer_installed:
            raise RuntimeError("evidence tokenizer is not installed")
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(self.mtc, embedding_name)
            for block in embedding.blocks:
                if self.tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT:
                    nn.init.zeros_(block.saliency_scale)
                else:
                    nn.init.zeros_(block.channel_gain)

    def zero_init_relay_gates(self) -> None:
        """Zero only NER gates without changing the trained tokenizer."""

        if not self.relay_enabled:
            if hasattr(self, "tpd_ner"):
                raise RuntimeError("relay-off unexpectedly registers tpd_ner")
            return
        if not self._nested_relay_installed:
            raise RuntimeError("relay is enabled but not installed")
        self.tpd_ner.zero_init_gates()

    def architecture_manifest(self) -> Dict[str, Any]:
        if not self._evidence_tokenizer_installed:
            raise RuntimeError("evidence tokenizer is not installed")
        if self.relay_enabled and not self._nested_relay_installed:
            raise RuntimeError("relay is enabled but not installed")
        is_v7_dch = (
            self.tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT
        )
        return {
            "model": self.__class__.__name__,
            "tokenizer_variant": self.tokenizer_variant,
            "primary_module": (
                "Keep-Context-Saliency V7-DCH"
                if is_v7_dch
                else "Capacity-matched progressive convolution control"
            ),
            "secondary_module": (
                "Nested Evidence Relay" if self.relay_enabled else None
            ),
            "embedding_replacements": ("embeddings_1", "embeddings_2"),
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "evidence_node_count": 5,
            "evidence_layout": (3, 2),
            "relay_enabled": self.relay_enabled,
            "relay_stage_order": RELAY_STAGE_ORDER,
            "relay_width": self.relay_width,
            "tensor_handoff": "forward_local_explicit_capability_checked",
            "semantic_sources": (
                ("Keep", "Context", "Saliency")
                if is_v7_dch
                else ("CapacityMatchedProgressiveConv",)
            ),
            "semantic_source_count": 3 if is_v7_dch else 1,
            "fourth_parallel_branch_added": False,
            "formal_launch_authorized": False,
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
        forward1 = getattr(embedding1, "forward_with_evidence", None)
        forward2 = getattr(embedding2, "forward_with_evidence", None)
        if not callable(forward1) or not callable(forward2):
            raise RuntimeError("evidence embedding capability is unavailable")
        emb1, evidence1 = forward1(x1)
        emb2, evidence2 = forward2(x2)
        emb3 = self.mtc.embeddings_3(x3)
        emb4 = self.mtc.embeddings_4(x4)
        if emb1 is None or emb2 is None or emb3 is None or emb4 is None:
            raise RuntimeError("an embedding unexpectedly returned None")
        if not isinstance(evidence1, tuple) or not isinstance(evidence2, tuple):
            raise RuntimeError("evidence interface must return tuples")
        if len(evidence1) != 3 or len(evidence2) != 2:
            raise RuntimeError("V7-DCH NER requires exactly 3+2 evidence nodes")
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
        if not isinstance(endpoint, torch.Tensor):
            raise RuntimeError(f"{name} endpoint is not a Tensor")
        if len(evidence) != expected_evidence:
            raise RuntimeError(
                f"{name} requires {expected_evidence} evidence tensors"
            )
        for index, tensor in enumerate(evidence):
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"{name} evidence[{index}] is not a Tensor")
            if tensor.dtype != source.dtype:
                raise RuntimeError(f"{name} evidence[{index}] dtype differs")
            if tensor.device != source.device:
                raise RuntimeError(f"{name} evidence[{index}] device differs")
        if endpoint.dtype != source.dtype:
            raise RuntimeError(f"{name} endpoint dtype differs")
        if endpoint.device != source.device:
            raise RuntimeError(f"{name} endpoint device differs")
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
        endpoint_shape = (
            batch,
            channels,
            height // (2 ** (expected_evidence + 1)),
            width // (2 ** (expected_evidence + 1)),
        )
        actual_shapes = [tuple(tensor.shape) for tensor in evidence]
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"{name} evidence shapes={actual_shapes}, "
                f"expected={expected_shapes}"
            )
        if tuple(endpoint.shape) != endpoint_shape:
            raise RuntimeError(
                f"{name} endpoint shape={tuple(endpoint.shape)}, "
                f"expected={endpoint_shape}"
            )

    def parameter_name_groups(self) -> Dict[str, Tuple[str, ...]]:
        """Return disjoint auditable backbone/tokenizer/relay name groups."""

        return {
            group: tuple(name for name, _ in entries)
            for group, entries in self.parameter_groups().items()
        }

    def parameter_groups(
        self,
    ) -> Dict[str, Tuple[Tuple[str, nn.Parameter], ...]]:
        """Return complete disjoint named-parameter groups for optimization."""

        tokenizer_prefixes = (
            "mtc.embeddings_1.",
            "mtc.embeddings_2.",
        )
        relay_prefix = "tpd_ner."
        named_parameters = tuple(self.named_parameters())
        parameter_ids = [id(parameter) for _, parameter in named_parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("named parameter traversal contains duplicates")
        grouped: Dict[str, list[Tuple[str, nn.Parameter]]] = {
            "backbone": [],
            "tokenizer": [],
            "relay": [],
        }
        for name, parameter in named_parameters:
            if name.startswith(relay_prefix):
                grouped["relay"].append((name, parameter))
            elif name.startswith(tokenizer_prefixes):
                grouped["tokenizer"].append((name, parameter))
            else:
                grouped["backbone"].append((name, parameter))
        flattened = tuple(
            entry
            for group in ("backbone", "tokenizer", "relay")
            for entry in grouped[group]
        )
        if {name for name, _ in flattened} != {
            name for name, _ in named_parameters
        }:
            raise RuntimeError("parameter groups do not cover the complete model")
        if self.relay_enabled != bool(grouped["relay"]):
            raise RuntimeError("relay parameter group contradicts relay_enabled")
        return {group: tuple(entries) for group, entries in grouped.items()}

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
            raise RuntimeError("V7-DCH NER evidence tokenizer is not installed")
        if not self.relay_enabled:
            return SCTransNet.forward(self, x)
        if not self._nested_relay_installed:
            raise RuntimeError("V7-DCH NER relay installation is incomplete")
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
    "EvidencePatchEmbeddingProtocol",
    "CapacityMatchedProgressiveBlock",
    "CapacityMatchedProgressivePatchEmbedding",
    "TPDNERV7DCHSCTransNet",
    "relay_parameter_count",
]
