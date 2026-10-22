"""The shared trunk and the decision head.

The trunk is LFM2.5-Encoder-350M: LFM2 blocks (10 short convolutions, 6 GQA attention layers) made bidirectional,
with a centred 3-tap convolution and no causal mask. Module names follow Hugging Face's LFM2.

An image or audio clip enters as a prefix of embeddings in front of the text. Prefix positions attend only to the
prefix and the convolution never lets them read the text, so the prefix is a function of the media alone; text
positions read everything. Text, image and audio calls run the same weights.

The head adds a question-type embedding, runs two pre-norm transformer layers over the text positions, and scores
the hidden state at each option marker with one shared MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.heads, self.kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.head_dim = d // self.heads
        self.q_proj = nn.Linear(d, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])
        self.k_layernorm = RMSNorm(self.head_dim, cfg["norm_eps"])

    def forward(self, x, cos, sin, mask):
        b, length, _ = x.shape
        shape = (b, length, -1, self.head_dim)
        q = self.q_layernorm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
        groups = self.heads // self.kv_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        return self.out_proj(y.transpose(1, 2).reshape(b, length, -1))


class ShortConv(nn.Module):
    """`out(C * conv(B * x))` with a centred 3-tap depthwise convolution."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden_size"]
        self.conv = nn.Conv1d(d, d, cfg["conv_L_cache"], groups=d, bias=False, padding=cfg["conv_L_cache"] - 1)
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, pad, keep_right):
        b, c, u = self.in_proj(x * pad[:, :, None]).transpose(-1, -2).chunk(3, dim=-2)
        bx = b * u
        w = self.conv.weight[:, 0, :]
        length = bx.shape[-1]
        xp = F.pad(bx, (1, 1))
        right = xp[..., 2:2 + length] * keep_right[:, None, :]  # media never reads the text
        y = xp[..., 0:length] * w[:, 0, None]
        y = y + xp[..., 1:1 + length] * w[:, 1, None]
        y = y + right * w[:, 2, None]
        return self.out_proj((c * y).transpose(-1, -2))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(2 * cfg["intermediate_size"] / 3)
        hidden = int(cfg["block_ffn_dim_multiplier"] * hidden)
        hidden = cfg["block_multiple_of"] * ((hidden + cfg["block_multiple_of"] - 1) // cfg["block_multiple_of"])
        d = cfg["hidden_size"]
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, cfg, kind: str):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(cfg)
        else:
            self.conv = ShortConv(cfg)
        self.feed_forward = MLP(cfg)
        self.operator_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])


class Trunk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(Layer(cfg, kind) for kind in cfg["layer_types"])
        self.embedding_norm = RMSNorm(cfg["hidden_size"], cfg["norm_eps"])
        self.rope_theta = cfg["rope_theta"]
        self.head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    def rope(self, length: int, x: torch.Tensor):
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim
        inv_freq = (1.0 / (self.rope_theta**exponent)).to(x.device)  # on the CPU, as HF's LFM2 builds it
        positions = torch.arange(length, device=x.device).float()
        freqs = (inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]

    def forward(self, h: torch.Tensor, pad: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """`h` (B, L, D) embeddings, `pad` (B, L) True on real tokens, `prefix` (B,) media lengths (0 for text)."""
        batch, length, _ = h.shape
        t = torch.arange(length, device=h.device)[None]
        media_query = t < prefix[:, None]
        text_key = t >= prefix[:, None]
        neg = max(NEG, torch.finfo(h.dtype).min)  # -1e9 does not fit in float16
        mask = torch.zeros(batch, 1, length, length, device=h.device, dtype=h.dtype)
        mask = mask + ((~pad).float() * neg).to(h.dtype)[:, None, None, :]
        mask = mask.masked_fill((media_query[:, :, None] & text_key[:, None, :])[:, None], neg)
        keep_right = (t != prefix[:, None] - 1).to(h.dtype)
        padf = pad.to(h.dtype)
        cos, sin = self.rope(length, h)
        for layer in self.layers:
            x = layer.operator_norm(h)
            if layer.is_attention_layer:
                x = layer.self_attn(x, cos, sin, mask)
            else:
                x = layer.conv(x, padf, keep_right)
            h = h + x
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return self.embedding_norm(h)


class DecisionHead(nn.Module):
    def __init__(self, d: int, layers: int):
        super().__init__()
        self.type_emb = nn.Embedding(3, d)
        layer = nn.TransformerEncoderLayer(d, d // 64, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h, pad, marker_pos, marker_mask, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=~pad)
        g = torch.gather(h, 1, marker_pos[:, :, None].expand(-1, -1, h.size(-1)))
        return self.scorer(g).squeeze(-1).float().masked_fill(~marker_mask, -1e4)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
