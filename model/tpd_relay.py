"""Nested target-evidence relay for TPD-SCTransNet.

This module is deliberately isolated from ``model.tpd``, ``model.tpd_clean``,
and ``model.SCTransNet``.  It upgrades an already constructed SCTransNet
instance by:

1. tapping the intermediate outputs that already exist inside embeddings_1/2;
2. building a narrow q4 -> q3 -> q2 evidence relay; and
3. applying zero-initialized spatial residual gates after decoder CCA.

The public SCTransNet forward contract is unchanged.  With all relay gates at
zero, installing the relay is numerically identical to the unmodified model.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_clean import CleanTPD2, CleanTPDPatchEmbedding


SpatialSize = Tuple[int, int]


class RelayRuntime:
    """Per-forward tensor handoff between embedding taps and decoder stages.

    The runtime is intentionally not an ``nn.Module``: it owns no parameters
    and is shared by the two taps and the registered relay module.  Tensor
    references are released after stage 2; only shape metadata is retained.
    """

    def __init__(self) -> None:
        self._active = False
        self._evidence: Dict[str, Tuple[torch.Tensor, ...]] = {}
        self._relay: Dict[int, torch.Tensor] = {}
        self.last_shapes: Dict[str, Tuple[int, ...]] = {}

    def begin(self) -> None:
        self._active = True
        self._evidence.clear()
        self._relay.clear()
        self.last_shapes.clear()

    def capture(self, key: str, values: Sequence[torch.Tensor]) -> None:
        if not self._active:
            raise RuntimeError("relay runtime has not started a forward pass")
        captured = tuple(values)
        self._evidence[key] = captured
        for index, value in enumerate(captured, start=1):
            self.last_shapes[f"{key}.h{index}"] = tuple(value.shape)

    def evidence(self, key: str) -> Tuple[torch.Tensor, ...]:
        try:
            return self._evidence[key]
        except KeyError as exc:
            raise RuntimeError(f"missing relay evidence for {key!r}") from exc

    def set_relay(self, stage: int, value: torch.Tensor) -> None:
        if not self._active:
            raise RuntimeError("relay runtime has not started a forward pass")
        self._relay[stage] = value
        self.last_shapes[f"q{stage}"] = tuple(value.shape)

    def relay(self, stage: int) -> torch.Tensor:
        try:
            return self._relay[stage]
        except KeyError as exc:
            raise RuntimeError(
                f"q{stage} is unavailable; relay stages must run q4 -> q3 -> q2"
            ) from exc

    def finish(self) -> None:
        self._evidence.clear()
        self._relay.clear()
        self._active = False

    def shape_snapshot(self) -> Dict[str, Tuple[int, ...]]:
        return dict(self.last_shapes)


class TPDEvidenceTap(nn.Module):
    """Expose intermediate block outputs while preserving embedding output.

    ``blocks`` is moved from the existing embedding rather than nesting that
    embedding under another module.  Therefore checkpoint keys remain
    ``mtc.embeddings_{1,2}.blocks.*``.
    """

    def __init__(
        self,
        embedding: nn.Module,
        key: str,
        runtime: RelayRuntime,
        *,
        expected_blocks: int,
    ) -> None:
        super().__init__()
        blocks = getattr(embedding, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(
                f"{key} must expose a ModuleList named 'blocks', got "
                f"{type(blocks).__name__}"
            )
        if len(blocks) != expected_blocks:
            raise ValueError(
                f"{key} requires {expected_blocks} blocks, got {len(blocks)}"
            )
        self.blocks = blocks
        self.key = key
        self.expected_blocks = expected_blocks
        self.runtime = runtime

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if self.key == "embeddings_1":
            self.runtime.begin()
        if x is None:
            return None
        intermediates = []
        for block in self.blocks:
            x = block(x)
            intermediates.append(x)
        self.runtime.capture(self.key, intermediates[:-1])
        return intermediates[-1]


class RelayFusionCell(nn.Module):
    """Project, align, and fuse a fixed set of evidence sources."""

    def __init__(self, source_channels: Sequence[int], width: int = 8) -> None:
        super().__init__()
        if width < 1:
            raise ValueError(f"relay width must be positive, got {width}")
        if not source_channels or any(channels < 1 for channels in source_channels):
            raise ValueError(f"invalid relay source channels: {source_channels}")
        self.width = width
        self.source_channels = tuple(int(channels) for channels in source_channels)
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, width, kernel_size=1, bias=False)
            for channels in self.source_channels
        )
        self.fuse = nn.Conv2d(
            len(self.source_channels) * width,
            width,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> torch.Tensor:
        if len(sources) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} relay sources, got {len(sources)}"
            )
        projected = []
        for index, (source, projection, expected_channels) in enumerate(
            zip(sources, self.projections, self.source_channels)
        ):
            if source.ndim != 4:
                raise ValueError(
                    f"relay source {index} must be BCHW, got shape={tuple(source.shape)}"
                )
            if source.shape[1] != expected_channels:
                raise ValueError(
                    f"relay source {index} requires C={expected_channels}, "
                    f"got C={source.shape[1]}"
                )
            value = projection(source)
            if value.shape[-2:] != output_size:
                value = F.interpolate(
                    value,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        return self.activation(self.fuse(torch.cat(projected, dim=1)))


class NestedEvidenceRelay(nn.Module):
    """Fixed q4 -> q3 -> q2 relay built from TPD intermediate evidence."""

    def __init__(
        self,
        runtime: RelayRuntime,
        *,
        base_channels: int = 32,
        width: int = 8,
    ) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError(
                f"base_channels must be positive, got {base_channels}"
            )
        self.runtime = runtime
        self.base_channels = base_channels
        self.width = width
        self.fusions = nn.ModuleDict(
            {
                "4": RelayFusionCell(
                    (base_channels, 2 * base_channels, 8 * base_channels),
                    width,
                ),
                "3": RelayFusionCell(
                    (
                        base_channels,
                        2 * base_channels,
                        width,
                        4 * base_channels,
                    ),
                    width,
                ),
                "2": RelayFusionCell(
                    (base_channels, width, 2 * base_channels),
                    width,
                ),
            }
        )
        self.gates = nn.ModuleDict(
            {
                str(stage): nn.Conv2d(width, 1, kernel_size=1)
                for stage in (4, 3, 2)
            }
        )
        self.zero_init_gates()

    def zero_init_gates(self) -> None:
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def stage_mask(self, stage: int, decoder_up: torch.Tensor) -> torch.Tensor:
        if stage not in (4, 3, 2):
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        if decoder_up.ndim != 4:
            raise ValueError(
                f"decoder feature must be BCHW, got {tuple(decoder_up.shape)}"
            )
        emb1 = self.runtime.evidence("embeddings_1")
        emb2 = self.runtime.evidence("embeddings_2")
        if len(emb1) != 3 or len(emb2) != 2:
            raise RuntimeError(
                "TPD-NER requires emb1/emb2 intermediate counts 3/2, got "
                f"{len(emb1)}/{len(emb2)}"
            )
        output_size = tuple(decoder_up.shape[-2:])
        if stage == 4:
            sources = (emb1[2], emb2[1], decoder_up)
        elif stage == 3:
            sources = (emb1[1], emb2[0], self.runtime.relay(4), decoder_up)
        else:
            sources = (emb1[0], self.runtime.relay(3), decoder_up)

        relay_value = self.fusions[str(stage)](sources, output_size)
        self.runtime.set_relay(stage, relay_value)
        mask = torch.tanh(self.gates[str(stage)](relay_value))
        self.runtime.last_shapes[f"mask{stage}"] = tuple(mask.shape)
        if stage == 2:
            self.runtime.finish()
        return mask


class RelayBinding:
    """Plain shared handle that keeps decoder wrappers copy-consistent."""

    def __init__(self, relay: NestedEvidenceRelay) -> None:
        self.relay = relay


class RelayUpBlock(nn.Module):
    """Existing CCA decoder block plus a TPD evidence spatial residual."""

    def __init__(
        self,
        *,
        up: nn.Module,
        coatt: nn.Module,
        nconvs: nn.Module,
        stage: int,
        binding: RelayBinding,
    ) -> None:
        super().__init__()
        if stage not in (4, 3, 2):
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        self.up = up
        self.coatt = coatt
        self.nConvs = nconvs
        self.stage = stage
        self.binding = binding

    @classmethod
    def from_existing(
        cls,
        block: nn.Module,
        *,
        stage: int,
        binding: RelayBinding,
    ) -> "RelayUpBlock":
        missing = [
            name for name in ("up", "coatt", "nConvs") if not hasattr(block, name)
        ]
        if missing:
            raise TypeError(
                f"decoder block is missing required attributes: {', '.join(missing)}"
            )
        return cls(
            up=block.up,
            coatt=block.coatt,
            nconvs=block.nConvs,
            stage=stage,
            binding=binding,
        )

    @property
    def relay(self) -> NestedEvidenceRelay:
        return self.binding.relay

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        if up.shape[-2:] != skip_x.shape[-2:]:
            up = F.interpolate(up, size=skip_x.shape[-2:], mode="nearest")
        skip_x_att = self.coatt(g=up, x=skip_x)
        mask = self.relay.stage_mask(self.stage, up)
        skip_x_att = skip_x_att * (1.0 + mask)
        return self.nConvs(torch.cat((skip_x_att, up), dim=1))


def _validate_full_tpd_embedding(
    embedding: nn.Module, name: str, expected_blocks: int
) -> None:
    if not isinstance(embedding, CleanTPDPatchEmbedding):
        raise TypeError(f"{name} must be a CleanTPDPatchEmbedding")
    if len(embedding.blocks) != expected_blocks:
        raise ValueError(f"{name} requires {expected_blocks} blocks")
    for index, block in enumerate(embedding.blocks):
        valid = (
            isinstance(block, CleanTPD2)
            and block.use_context is True
            and block.use_saliency is True
            and block.context_scale is not None
            and block.saliency_scale is not None
        )
        if not valid:
            raise TypeError(
                f"{name}.blocks[{index}] is not full Keep-Context-Saliency TPD"
            )


def _embedding_channels(embedding: nn.Module, name: str) -> int:
    blocks = getattr(embedding, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError(f"{name} must contain a non-empty ModuleList named 'blocks'")
    projection = getattr(blocks[0], "phase_compress", None)
    channels = getattr(projection, "out_channels", None)
    if not isinstance(channels, int) or channels < 1:
        raise TypeError(f"cannot infer channels from {name}.blocks[0]")
    return channels


def _validate_decoder_channels(
    block: nn.Module, name: str, expected_channels: int
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


def install_tpd_ner(
    model: nn.Module,
    replacements: Mapping[str, nn.Module],
    *,
    width: int = 8,
) -> Dict[str, nn.Module]:
    """Install TPD-NER without changing the SCTransNet forward interface.

    Install this only after the base model and its TPD embeddings have been
    initialized.  If a later initializer touches the whole model, call
    ``parts["relay"].zero_init_gates()`` again before training.
    """

    if hasattr(model, "tpd_ner"):
        raise ValueError("TPD-NER is already installed on this model")
    mtc = getattr(model, "mtc", None)
    if mtc is None:
        raise TypeError("model must expose an 'mtc' ChannelTransformer")
    embedding1 = getattr(mtc, "embeddings_1", None)
    embedding2 = getattr(mtc, "embeddings_2", None)
    for name, current in (
        ("embeddings_1", embedding1),
        ("embeddings_2", embedding2),
    ):
        supplied = replacements.get(name)
        if supplied is not current:
            raise ValueError(
                f"replacement {name!r} does not match the module installed in model"
            )

    _validate_full_tpd_embedding(embedding1, "embeddings_1", 4)
    _validate_full_tpd_embedding(embedding2, "embeddings_2", 3)

    base_channels = _embedding_channels(embedding1, "embeddings_1")
    second_channels = _embedding_channels(embedding2, "embeddings_2")
    if second_channels != 2 * base_channels:
        raise ValueError(
            "TPD-NER expects embeddings_2 channels to equal 2 * embeddings_1 "
            f"channels, got {second_channels} and {base_channels}"
        )

    _validate_decoder_channels(model.up_decoder4, "up_decoder4", 8 * base_channels)
    _validate_decoder_channels(model.up_decoder3, "up_decoder3", 4 * base_channels)
    _validate_decoder_channels(model.up_decoder2, "up_decoder2", 2 * base_channels)

    runtime = RelayRuntime()
    tap1 = TPDEvidenceTap(
        embedding1,
        "embeddings_1",
        runtime,
        expected_blocks=4,
    )
    tap2 = TPDEvidenceTap(
        embedding2,
        "embeddings_2",
        runtime,
        expected_blocks=3,
    )
    relay = NestedEvidenceRelay(
        runtime,
        base_channels=base_channels,
        width=width,
    )
    binding = RelayBinding(relay)

    decoder4 = RelayUpBlock.from_existing(
        model.up_decoder4,
        stage=4,
        binding=binding,
    )
    decoder3 = RelayUpBlock.from_existing(
        model.up_decoder3,
        stage=3,
        binding=binding,
    )
    decoder2 = RelayUpBlock.from_existing(
        model.up_decoder2,
        stage=2,
        binding=binding,
    )
    mtc.embeddings_1 = tap1
    mtc.embeddings_2 = tap2
    model.add_module("tpd_ner", relay)
    model.up_decoder4 = decoder4
    model.up_decoder3 = decoder3
    model.up_decoder2 = decoder2
    relay.zero_init_gates()
    return {
        "embedding_1": tap1,
        "embedding_2": tap2,
        "relay": relay,
        "decoder_4": model.up_decoder4,
        "decoder_3": model.up_decoder3,
        "decoder_2": model.up_decoder2,
    }


def relay_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
