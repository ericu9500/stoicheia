"""HF-Hub-compatible model for Stoicheia (CharBertEncoder).

Self-contained: vendors the transformer primitives (RMSNorm/RoPE/Attention/GeGLU/Block)
so this file has no dependency on the original research repo. Uses the SDPA attention
path only (portable to CPU and any CUDA GPU) -- the original training code also supports
a compiled FlexAttention block-sparse path for long packed sequences on GPU, which is not
needed for standalone inference on single passages and is omitted here for portability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_char_bert import CharBertConfig


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


class RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def cos_sin(self, pos):
        # Recomputed on every call rather than cached in a registered buffer: a
        # persistent=False buffer is never covered by the checkpoint's state dict,
        # so it depends entirely on __init__-time materialization -- which some
        # transformers versions' meta-device/low_cpu_mem_usage loading path can
        # skip, silently leaving this tensor uninitialized. Recomputing here is
        # immune to that regardless of how the model was constructed/loaded.
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=pos.device).float() / self.dim))
        f = torch.outer(pos.float(), inv)
        emb = torch.cat([f, f], -1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], -1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None]
    sin = sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, d, n_heads, rope: RoPE, qk_norm=False):
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = rope
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.dh)
            self.k_norm = RMSNorm(self.dh)

    def forward(self, x, pos, attn_mask):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self.rope.cos_sin(pos)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class GeGLU(nn.Module):
    def __init__(self, d, mult=8 / 3):
        super().__init__()
        hidden = int(d * mult)
        hidden = (hidden + 63) // 64 * 64
        self.wi = nn.Linear(d, 2 * hidden, bias=False)
        self.wo = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, -1)
        return self.wo(F.gelu(a) * b)


class Block(nn.Module):
    def __init__(self, d, n_heads, rope, window=0, qk_norm=False):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, rope, qk_norm=qk_norm)
        self.n2 = RMSNorm(d)
        self.mlp = GeGLU(d)
        self.window = window  # 0 = global; >0 = local sliding window (characters)

    def forward(self, x, pos, base_mask):
        x = x + self.attn(self.n1(x), pos, base_mask)
        x = x + self.mlp(self.n2(x))
        return x


def build_attn_mask(seg_id, window, device, dtype):
    """Additive mask (B,1,T,T): same-segment AND (window==0 or |i-j|<window)."""
    B, T = seg_id.shape
    same = seg_id[:, None, :] == seg_id[:, :, None]
    if window and window > 0:
        idx = torch.arange(T, device=device)
        near = (idx[None, :] - idx[:, None]).abs() < window
        same = same & near[None]
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    mask.masked_fill_(~same[:, None], float("-inf"))
    return mask


@dataclass
class CharBertOutput(ModelOutput):
    char: torch.FloatTensor = None
    boundary: torch.FloatTensor = None
    dia: torch.FloatTensor = None
    cap: torch.FloatTensor = None
    punct: torch.FloatTensor = None
    hidden_states: Optional[tuple] = None


class CharBertModel(PreTrainedModel):
    config_class = CharBertConfig

    def __init__(self, config: CharBertConfig):
        super().__init__(config)
        self.e_char = nn.Embedding(config.n_char_ids, config.d_model)
        self.e_bnd = nn.Embedding(config.n_boundary, config.d_model)
        self.e_dia = nn.Embedding(config.n_dia, config.d_model)
        self.e_punct = nn.Embedding(config.n_punct, config.d_model)
        rope = RoPE(config.d_model // config.n_heads)
        blocks = []
        for i in range(config.depth):
            win = 0 if i % 4 == 3 else config.char_window  # 3 local : 1 global
            blocks.append(Block(config.d_model, config.n_heads, rope, window=win, qk_norm=config.qk_norm))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = RMSNorm(config.d_model)
        self.head_char = nn.Linear(config.d_model, config.n_char_ids, bias=False)
        self.head_bnd = nn.Linear(config.d_model, 3, bias=False)
        self.head_dia = nn.Linear(config.d_model, 48, bias=False)
        self.head_cap = nn.Linear(config.d_model, 2, bias=False)
        self.head_punct = nn.Linear(config.d_model, 6, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        boundary: torch.LongTensor,
        dia: torch.LongTensor,
        punct: torch.LongTensor,
        seg_id: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        cfg = self.config
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        seg = seg_id if seg_id is not None else torch.zeros(B, T, dtype=torch.long, device=input_ids.device)

        x = self.e_char(input_ids) + self.e_bnd(boundary) + self.e_dia(dia) + self.e_punct(punct)
        attn_mask = build_attn_mask(seg, cfg.char_window, input_ids.device, x.dtype)
        glob_mask = build_attn_mask(seg, 0, input_ids.device, x.dtype)

        hidden_states = [] if output_hidden_states else None
        for blk in self.blocks:
            m = glob_mask if blk.window == 0 else attn_mask
            x = blk(x, pos, m)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm_out(x)
        out = dict(
            char=self.head_char(x),
            boundary=self.head_bnd(x),
            dia=self.head_dia(x),
            cap=self.head_cap(x),
            punct=self.head_punct(x),
        )
        if output_hidden_states:
            out["hidden_states"] = tuple(hidden_states) + (x,)
        if not return_dict:
            return tuple(v for v in out.values() if v is not None)
        return CharBertOutput(**out)
