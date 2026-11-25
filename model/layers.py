"""Shared transformer primitives (ModernBERT-style: pre-norm, RoPE, GeGLU, no bias).

Kept backend-agnostic: attention uses F.scaled_dot_product_attention, which runs on CPU
(bring-up / overfit tests) and dispatches to FlashAttention on CUDA. A varlen/FlexAttention
fast path is swapped in during MFU tuning; the math here is the reference.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)

    def cos_sin(self, pos):
        # pos: (T,) absolute positions
        f = torch.outer(pos.float(), self.inv)          # (T, dim/2)
        emb = torch.cat([f, f], -1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], -1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, Dh); cos,sin: (T, Dh)
    cos = cos[None, None]; sin = sin[None, None]
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
        if qk_norm:                              # per-head RMSNorm on q,k before RoPE (stabilizes grads)
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
        if attn_mask is None or isinstance(attn_mask, torch.Tensor):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # SDPA (CPU/GPU)
        else:                                   # FlexAttention BlockMask (block-sparse, O(T) mem)
            out = _flex(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)


# flex_attention must be explicitly wrapped in torch.compile to get its fused block-sparse
# Triton kernel -- called eagerly it silently falls back to math_attention, which
# materializes the full dense (B,H,T,T) score matrix and OOMs on any real-sized batch (see
# the warning torch itself prints when this is skipped). This has to happen here, at module
# import time in plain eager Python -- lazily compiling on first call doesn't work when that
# first call happens from inside an outer torch.compile(model) trace (finetune scripts wrap
# the whole model): invoking torch.compile() itself while dynamo is already tracing is a
# nested-compile pattern it can't honor, and it silently graph-breaks back to the
# uncompiled function, reproducing the exact same OOM.
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw

_flex_fn = torch.compile(_flex_attention_raw, dynamic=False)


def _flex(q, k, v, block_mask):
    return _flex_fn(q, k, v, block_mask=block_mask)


def build_block_mask(seg_id, window, device):
    """FlexAttention BlockMask: attend within same doc AND (global or |i-j|<window)."""
    from torch.nn.attention.flex_attention import create_block_mask
    B, T = seg_id.shape

    def mask_mod(b, h, qi, ki):
        same = seg_id[b, qi] == seg_id[b, ki]
        if window and window > 0:
            return same & ((qi - ki).abs() < window)
        return same

    return create_block_mask(mask_mod, B, None, T, T, device=device, _compile=True)


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
        self.window = window        # 0 = global; >0 = local sliding window (chars)

    def forward(self, x, pos, base_mask):
        x = x + self.attn(self.n1(x), pos, base_mask)
        x = x + self.mlp(self.n2(x))
        return x


def build_attn_mask(seg_id, window, device, dtype):
    """Additive mask (B,1,T,T): same-segment AND (window==0 or |i-j|<window)."""
    B, T = seg_id.shape
    same = seg_id[:, None, :] == seg_id[:, :, None]         # (B,T,T)
    if window and window > 0:
        idx = torch.arange(T, device=device)
        near = (idx[None, :] - idx[:, None]).abs() < window
        same = same & near[None]
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    mask.masked_fill_(~same[:, None], float("-inf"))
    return mask
