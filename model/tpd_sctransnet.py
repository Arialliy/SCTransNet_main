"""First-class TPD-SCTransNet with explicit five-node evidence relay.

The tokenizer keeps the original Keep-Context-Saliency mainline.  Its five
intermediate nodes ``h11/h12/h13/h21/h22`` are passed as ordinary local
tensors through ``q4 -> q3 -> q2``; no hook, tensor cache, or shared runtime
object participates in forward propagation.

Parameter module names intentionally match the earlier installer-based NER so
that its state dict remains strictly loadable into this implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean import CleanTPD2, CleanTPDPatchEmbedding
from model.tpd_relay import RelayFusionCell


SpatialSize = Tuple[int, int]
EVIDENCE_NODE_NAMES = ("h11", "h12", "h13", "h21", "h22")
RELAY_STAGE_ORDER = (4, 3, 2)


def _validate_full_embedding(
    embedding: nn.Module,
    name: str,
    expected_blocks: int,
) -> None:
    if not isinstance(embedding, CleanTPDPatchEmbedding):
        raise TypeError(f"{name} must be a CleanTPDPatchEmbedding")
    if len(embedding.blocks) != expected_blocks:
        raise ValueError(
            f"{name} requires {expected_blocks} blocks, "
            f"got {len(embedding.blocks)}"
        )
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
        raise TypeError(f"{name} must expose a non-empty ModuleList named 'blocks'")
    projection = getattr(blocks[0], "phase_compress", None)
    channels = getattr(projection, "out_channels", None)
    if not isinstance(channels, int) or channels < 1:
        raise TypeError(f"cannot infer channels from {name}.blocks[0]")
    return channels


def _original_embedding_contract(
    embedding: nn.Module,
    name: str,
) -> Tuple[int, int]:
    projection = getattr(embedding, "patch_embeddings", None)
    channels = getattr(projection, "in_channels", None)
    stride = getattr(projection, "stride", None)
    if not isinstance(channels, int) or channels < 1:
        raise TypeError(f"cannot infer input channels from original {name}")
    if not isinstance(stride, tuple) or len(stride) != 2 or stride[0] != stride[1]:
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


class ExplicitTPDEvidenceEmbedding(nn.Module):
    """Full TPD embedding that explicitly returns its intermediate nodes."""

    def __init__(
        self,
        embedding: CleanTPDPatchEmbedding,
        *,
        name: str,
        expected_blocks: int,
    ) -> None:
        super().__init__()
        _validate_full_embedding(embedding, name, expected_blocks)
        # Transfer the ModuleList directly to preserve checkpoint key names.
        self.blocks = embedding.blocks
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


class ExplicitNestedEvidenceRelay(nn.Module):
    """Narrow q4 -> q3 -> q2 relay with no persistent tensor state."""

    def __init__(self, *, base_channels: int = 32, width: int = 8) -> None:
        super().__init__()
        if base_channels < 1 or width < 1:
            raise ValueError("base_channels and width must be positive")
        self.base_channels = int(base_channels)
        self.width = int(width)
        self.fusions = nn.ModuleDict(
            {
                "4": RelayFusionCell(
                    (
                        self.base_channels,
                        2 * self.base_channels,
                        8 * self.base_channels,
                    ),
                    self.width,
                ),
                "3": RelayFusionCell(
                    (
                        self.base_channels,
                        2 * self.base_channels,
                        self.width,
                        4 * self.base_channels,
                    ),
                    self.width,
                ),
                "2": RelayFusionCell(
                    (
                        self.base_channels,
                        self.width,
                        2 * self.base_channels,
                    ),
                    self.width,
                ),
            }
        )
        self.gates = nn.ModuleDict(
            {
                str(stage): nn.Conv2d(self.width, 1, kernel_size=1)
                for stage in RELAY_STAGE_ORDER
            }
        )
        self.zero_init_gates()

    def zero_init_gates(self) -> None:
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            if gate.bias is not None:
                nn.init.zeros_(gate.bias)

    def forward_stage(
        self,
        stage: int,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if stage not in RELAY_STAGE_ORDER:
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError(f"invalid relay output size: {output_size}")
        relay_value = self.fusions[str(stage)](sources, output_size)
        return relay_value, torch.tanh(self.gates[str(stage)](relay_value))


class ExplicitRelayUpBlock(nn.Module):
    """CCA decoder block whose TPD mask is supplied explicitly."""

    def __init__(
        self,
        *,
        up: nn.Module,
        coatt: nn.Module,
        nconvs: nn.Module,
        stage: int,
    ) -> None:
        super().__init__()
        if stage not in RELAY_STAGE_ORDER:
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")
        self.up = up
        self.coatt = coatt
        self.nConvs = nconvs
        self.stage = stage

    @classmethod
    def from_existing(
        cls,
        block: nn.Module,
        *,
        stage: int,
    ) -> "ExplicitRelayUpBlock":
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
        )

    def prepare(
        self,
        x: torch.Tensor,
        skip_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        up = self.up(x)
        if up.shape[-2:] != skip_x.shape[-2:]:
            up = F.interpolate(up, size=skip_x.shape[-2:], mode="nearest")
        return up, self.coatt(g=up, x=skip_x)

    def finish(
        self,
        up: torch.Tensor,
        skip_x_att: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (skip_x_att.shape[0], 1, *skip_x_att.shape[-2:])
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"stage-{self.stage} mask shape={tuple(mask.shape)}, "
                f"expected={expected}"
            )
        if up.shape[-2:] != skip_x_att.shape[-2:]:
            raise ValueError(f"stage-{self.stage} up/skip spatial mismatch")
        skip_x_att = skip_x_att * (1.0 + mask)
        return self.nConvs(torch.cat((skip_x_att, up), dim=1))


class TPDSCTransNet(SCTransNet):
    """SCTransNet backbone with full TPD tokenization and explicit NER.

    ``install_tpd=False`` is used by the paired experiment builder so the
    shared backbone can be initialized before candidate-specific modules.
    """

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
        relay_width: int = 8,
        install_tpd: bool = True,
    ) -> None:
        super().__init__(
            config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
        )
        if relay_width < 1:
            raise ValueError(f"relay_width must be positive, got {relay_width}")
        self.relay_width = int(relay_width)
        self._tpd_tokenizer_installed = False
        self._tpd_relay_installed = False
        if install_tpd:
            self.install_tpd_tokenizer()
            self.install_nested_relay()

    def install_tpd_tokenizer(self) -> Dict[str, ExplicitTPDEvidenceEmbedding]:
        if self._tpd_tokenizer_installed:
            raise ValueError("full TPD tokenizer is already installed")
        if self._tpd_relay_installed or hasattr(self, "tpd_ner"):
            raise RuntimeError("cannot replace tokenizer after relay installation")
        channels1, stride1 = _original_embedding_contract(
            self.mtc.embeddings_1, "embeddings_1"
        )
        channels2, stride2 = _original_embedding_contract(
            self.mtc.embeddings_2, "embeddings_2"
        )
        if channels2 != 2 * channels1:
            raise ValueError("embeddings_2 channels must equal 2 * embeddings_1")
        raw1 = CleanTPDPatchEmbedding(
            channels1, stride1, use_context=True, use_saliency=True
        )
        raw2 = CleanTPDPatchEmbedding(
            channels2, stride2, use_context=True, use_saliency=True
        )
        embedding1 = ExplicitTPDEvidenceEmbedding(
            raw1, name="embeddings_1", expected_blocks=4
        )
        embedding2 = ExplicitTPDEvidenceEmbedding(
            raw2, name="embeddings_2", expected_blocks=3
        )
        self.mtc.embeddings_1 = embedding1
        self.mtc.embeddings_2 = embedding2
        self._tpd_tokenizer_installed = True
        return {
            "embeddings_1": embedding1,
            "embeddings_2": embedding2,
        }

    def install_nested_relay(self) -> Dict[str, nn.Module]:
        if not self._tpd_tokenizer_installed:
            raise RuntimeError("install the full TPD tokenizer before NER")
        if self._tpd_relay_installed or hasattr(self, "tpd_ner"):
            raise ValueError("Nested Evidence Relay is already installed")
        embedding1 = self.mtc.embeddings_1
        embedding2 = self.mtc.embeddings_2
        if not isinstance(embedding1, ExplicitTPDEvidenceEmbedding):
            raise TypeError("embeddings_1 is not explicit full TPD")
        if not isinstance(embedding2, ExplicitTPDEvidenceEmbedding):
            raise TypeError("embeddings_2 is not explicit full TPD")
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
        self._tpd_relay_installed = True
        relay.zero_init_gates()
        return {
            "embedding_1": embedding1,
            "embedding_2": embedding2,
            "relay": relay,
            "decoder_4": decoder4,
            "decoder_3": decoder3,
            "decoder_2": decoder2,
        }

    def zero_init_target_residuals(self) -> None:
        if not self._tpd_tokenizer_installed or not self._tpd_relay_installed:
            raise RuntimeError("TPD tokenizer and NER must both be installed")
        for name in ("embeddings_1", "embeddings_2"):
            for block in getattr(self.mtc, name).blocks:
                nn.init.zeros_(block.context_scale)
                nn.init.zeros_(block.saliency_scale)
        self.tpd_ner.zero_init_gates()

    def architecture_manifest(self) -> Dict[str, Any]:
        if not self._tpd_tokenizer_installed or not self._tpd_relay_installed:
            raise RuntimeError("TPD tokenizer and NER must both be installed")
        return {
            "model": "TPDSCTransNet",
            "primary_module": "Keep-Context-Saliency TPD",
            "secondary_module": "Nested Evidence Relay",
            "embedding_replacements": ("embeddings_1", "embeddings_2"),
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "relay_stage_order": RELAY_STAGE_ORDER,
            "relay_width": self.relay_width,
            "tensor_handoff": "forward_local_explicit",
        }

    def _explicit_embeddings(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ):
        embedding1 = self.mtc.embeddings_1
        embedding2 = self.mtc.embeddings_2
        if not isinstance(embedding1, ExplicitTPDEvidenceEmbedding):
            raise RuntimeError("explicit embeddings_1 is not installed")
        if not isinstance(embedding2, ExplicitTPDEvidenceEmbedding):
            raise RuntimeError("explicit embeddings_2 is not installed")
        emb1, evidence1 = embedding1.forward_with_evidence(x1)
        emb2, evidence2 = embedding2.forward_with_evidence(x2)
        emb3 = self.mtc.embeddings_3(x3)
        emb4 = self.mtc.embeddings_4(x4)
        if emb3 is None or emb4 is None:
            raise RuntimeError("embeddings_3/4 unexpectedly returned None")
        if len(evidence1) != 3 or len(evidence2) != 2:
            raise RuntimeError("TPD-SCTransNet requires exactly 3+2 evidence nodes")
        return emb1, emb2, emb3, emb4, evidence1, evidence2

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        if not self._tpd_tokenizer_installed or not self._tpd_relay_installed:
            raise RuntimeError("TPDSCTransNet tokenizer/relay installation incomplete")
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))
        f1, f2, f3, f4 = x1, x2, x3, x4

        emb1, emb2, emb3, emb4, evidence1, evidence2 = (
            self._explicit_embeddings(x1, x2, x3, x4)
        )
        h11, h12, h13 = evidence1
        h21, h22 = evidence2
        encoded1, encoded2, encoded3, encoded4, _ = self.mtc.encoder(
            emb1, emb2, emb3, emb4
        )
        x1 = self.mtc.reconstruct_1(encoded1) + f1
        x2 = self.mtc.reconstruct_2(encoded2) + f2
        x3 = self.mtc.reconstruct_3(encoded3) + f3
        x4 = self.mtc.reconstruct_4(encoded4) + f4
        x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

        up4, skip4 = self.up_decoder4.prepare(d5, x4)
        q4, mask4 = self.tpd_ner.forward_stage(
            4, (h13, h22, up4), tuple(up4.shape[-2:])
        )
        d4 = self.up_decoder4.finish(up4, skip4, mask4)
        up3, skip3 = self.up_decoder3.prepare(d4, x3)
        q3, mask3 = self.tpd_ner.forward_stage(
            3, (h12, h21, q4, up3), tuple(up3.shape[-2:])
        )
        d3 = self.up_decoder3.finish(up3, skip3, mask3)
        up2, skip2 = self.up_decoder2.prepare(d3, x2)
        _, mask2 = self.tpd_ner.forward_stage(
            2, (h11, q3, up2), tuple(up2.shape[-2:])
        )
        d2 = self.up_decoder2.finish(up2, skip2, mask2)
        out = self.outc(self.up_decoder1(d2, x1))

        if not self.deepsuper:
            return torch.sigmoid(out)
        gt_5 = self.gt_conv5(d5)
        gt_4 = self.gt_conv4(d4)
        gt_3 = self.gt_conv3(d3)
        gt_2 = self.gt_conv2(d2)
        gt5 = F.interpolate(gt_5, scale_factor=16, mode="bilinear", align_corners=True)
        gt4 = F.interpolate(gt_4, scale_factor=8, mode="bilinear", align_corners=True)
        gt3 = F.interpolate(gt_3, scale_factor=4, mode="bilinear", align_corners=True)
        gt2 = F.interpolate(gt_2, scale_factor=2, mode="bilinear", align_corners=True)
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


def tpd_sctransnet_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
