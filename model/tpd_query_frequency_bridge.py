"""Pure-function bridge from SCTransNet attention to Query-only FG.

The three functions in this module deliberately mirror the forward methods of
``Attention_org``, ``Block_ViT``, and ``Encoder`` from ``model.SCTransNet``.
They do not wrap, replace, or register any module.  The only additional
operation is one call to ``qfg.apply_prepared`` after the four Query
convolutions and before Query rearrangement/normalization.

The prepared frequency object is an explicit forward-local argument all the
way through the call stack.  Consequently, it is reusable by every SCTB in an
Encoder forward without being stored on the Encoder, block, attention, or
frequency-gate module.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from einops import rearrange

from model.tpd_frequency_gate import (
    PreparedQueryFrequencyGate,
    QueryOnlyFrequencyGate,
)


AttentionResult = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    None,
]
EncoderResult = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list,
]


def frequency_attention_forward(
    attention,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    emb3: torch.Tensor,
    emb4: torch.Tensor,
    emb_all: torch.Tensor,
    qfg: QueryOnlyFrequencyGate,
    prepared: PreparedQueryFrequencyGate,
) -> AttentionResult:
    """Run ``Attention_org`` with Query-only frequency modulation.

    This is the original ``Attention_org.forward`` operation order with one
    insertion: ``qfg.apply_prepared`` receives only ``q1...q4`` after their
    pointwise and grouped spatial convolutions.  K and V remain on their
    original path.
    """

    _batch, _channels, height, width = emb1.shape
    q1 = attention.q1(attention.mhead1(emb1))
    q2 = attention.q2(attention.mhead2(emb2))
    q3 = attention.q3(attention.mhead3(emb3))
    q4 = attention.q4(attention.mhead4(emb4))
    k = attention.k(attention.mheadk(emb_all))
    v = attention.v(attention.mheadv(emb_all))

    gated = qfg.apply_prepared((q1, q2, q3, q4), prepared)
    q1, q2, q3, q4 = gated.queries

    q1 = rearrange(
        q1,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )
    q2 = rearrange(
        q2,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )
    q3 = rearrange(
        q3,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )
    q4 = rearrange(
        q4,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )
    k = rearrange(
        k,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )
    v = rearrange(
        v,
        "b (head c) h w -> b head c (h w)",
        head=attention.num_attention_heads,
    )

    q1 = torch.nn.functional.normalize(q1, dim=-1)
    q2 = torch.nn.functional.normalize(q2, dim=-1)
    q3 = torch.nn.functional.normalize(q3, dim=-1)
    q4 = torch.nn.functional.normalize(q4, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    attn1 = (q1 @ k.transpose(-2, -1)) / math.sqrt(attention.KV_size)
    attn2 = (q2 @ k.transpose(-2, -1)) / math.sqrt(attention.KV_size)
    attn3 = (q3 @ k.transpose(-2, -1)) / math.sqrt(attention.KV_size)
    attn4 = (q4 @ k.transpose(-2, -1)) / math.sqrt(attention.KV_size)

    attention_probs1 = attention.softmax(attention.psi(attn1))
    attention_probs2 = attention.softmax(attention.psi(attn2))
    attention_probs3 = attention.softmax(attention.psi(attn3))
    attention_probs4 = attention.softmax(attention.psi(attn4))

    out1 = attention_probs1 @ v
    out2 = attention_probs2 @ v
    out3 = attention_probs3 @ v
    out4 = attention_probs4 @ v

    out_1 = out1.mean(dim=1)
    out_2 = out2.mean(dim=1)
    out_3 = out3.mean(dim=1)
    out_4 = out4.mean(dim=1)

    out_1 = rearrange(
        out_1,
        "b c (h w) -> b c h w",
        h=height,
        w=width,
    )
    out_2 = rearrange(
        out_2,
        "b c (h w) -> b c h w",
        h=height,
        w=width,
    )
    out_3 = rearrange(
        out_3,
        "b c (h w) -> b c h w",
        h=height,
        w=width,
    )
    out_4 = rearrange(
        out_4,
        "b c (h w) -> b c h w",
        h=height,
        w=width,
    )

    output1 = attention.project_out1(out_1)
    output2 = attention.project_out2(out_2)
    output3 = attention.project_out3(out_3)
    output4 = attention.project_out4(out_4)
    weights = None
    return output1, output2, output3, output4, weights


def frequency_block_forward(
    block,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    emb3: torch.Tensor,
    emb4: torch.Tensor,
    qfg: QueryOnlyFrequencyGate,
    prepared: PreparedQueryFrequencyGate,
) -> AttentionResult:
    """Run one ``Block_ViT`` while explicitly forwarding the prepared FG."""

    embcat = []
    org1 = emb1
    org2 = emb2
    org3 = emb3
    org4 = emb4
    for embedding in (emb1, emb2, emb3, emb4):
        if embedding is not None:
            embcat.append(embedding)
    emb_all = torch.cat(embcat, dim=1)

    cx1 = block.attn_norm1(emb1) if emb1 is not None else None
    cx2 = block.attn_norm2(emb2) if emb2 is not None else None
    cx3 = block.attn_norm3(emb3) if emb3 is not None else None
    cx4 = block.attn_norm4(emb4) if emb4 is not None else None
    emb_all = block.attn_norm(emb_all)
    cx1, cx2, cx3, cx4, weights = frequency_attention_forward(
        block.channel_attn,
        cx1,
        cx2,
        cx3,
        cx4,
        emb_all,
        qfg,
        prepared,
    )

    cx1 = org1 + cx1 if emb1 is not None else None
    cx2 = org2 + cx2 if emb2 is not None else None
    cx3 = org3 + cx3 if emb3 is not None else None
    cx4 = org4 + cx4 if emb4 is not None else None

    org1 = cx1
    org2 = cx2
    org3 = cx3
    org4 = cx4
    x1 = block.ffn_norm1(cx1) if emb1 is not None else None
    x2 = block.ffn_norm2(cx2) if emb2 is not None else None
    x3 = block.ffn_norm3(cx3) if emb3 is not None else None
    x4 = block.ffn_norm4(cx4) if emb4 is not None else None
    x1 = block.ffn1(x1) if emb1 is not None else None
    x2 = block.ffn2(x2) if emb2 is not None else None
    x3 = block.ffn3(x3) if emb3 is not None else None
    x4 = block.ffn4(x4) if emb4 is not None else None
    x1 = x1 + org1 if emb1 is not None else None
    x2 = x2 + org2 if emb2 is not None else None
    x3 = x3 + org3 if emb3 is not None else None
    x4 = x4 + org4 if emb4 is not None else None

    return x1, x2, x3, x4, weights


def frequency_encoder_forward(
    encoder,
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    emb3: torch.Tensor,
    emb4: torch.Tensor,
    qfg: QueryOnlyFrequencyGate,
    prepared: PreparedQueryFrequencyGate,
) -> EncoderResult:
    """Run ``Encoder`` with one explicit prepared FG shared by every SCTB."""

    attn_weights = []
    for layer_block in encoder.layer:
        emb1, emb2, emb3, emb4, weights = frequency_block_forward(
            layer_block,
            emb1,
            emb2,
            emb3,
            emb4,
            qfg,
            prepared,
        )
        if encoder.vis:
            attn_weights.append(weights)
    emb1 = encoder.encoder_norm1(emb1) if emb1 is not None else None
    emb2 = encoder.encoder_norm2(emb2) if emb2 is not None else None
    emb3 = encoder.encoder_norm3(emb3) if emb3 is not None else None
    emb4 = encoder.encoder_norm4(emb4) if emb4 is not None else None
    return emb1, emb2, emb3, emb4, attn_weights


__all__ = [
    "frequency_attention_forward",
    "frequency_block_forward",
    "frequency_encoder_forward",
]
