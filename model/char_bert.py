"""Stoicheia: flat character-level masked-diffusion encoder.

Reads scriptio continua (24-letter minimal Greek alphabet, ς->σ, no accents) plus four
parallel per-character channels (word-boundary, diacritics, capitalization, punctuation),
each independently maskable. ModernBERT-style blocks (pre-norm, RoPE, GeGLU, QK-norm, no
bias) alternating local/global attention. No subword trunk, no router, no hierarchy —
that design was tried and rejected (see the project history): a hierarchical arm's
apparent quality edge traced to a routing leak (masked words kept their true segmentation
at train time, unavailable at inference), and on every leak-free metric the flat model won.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.layers import Block, RMSNorm, RoPE, build_attn_mask, build_block_mask


@dataclass
class CharBertConfig:
    # vocab / channels
    n_alpha: int = 24
    mask_id: int = 24
    blank_id: int = 25          # ∅ : "gap ends / no char here" (elastic unknown-length gaps)
    pad_id: int = 26
    n_char_ids: int = 27        # alpha + mask + blank + pad
    n_boundary: int = 4         # 0 internal, 1 word-final, 2 sent-final, 3 unknown
    n_dia: int = 49             # 48 accent/breathing/iota states + unknown
    n_punct: int = 7            # 6 punctuation classes + unknown
    # optional metadata conditioning (region / date), off by default (0) so every
    # existing checkpoint loads unchanged; only insc metadata-primed fine-tunes set these
    n_region: int = 0
    n_century: int = 0
    # dims
    d_model: int = 1024
    n_heads: int = 16
    depth: int = 32
    char_window: int = 256      # local-attention window; every 4th block is global (3:1)
    attn_impl: str = "flex"     # "flex" (compiled block-sparse, GPU, long seq) | "sdpa" (dense, CPU/short)
    qk_norm: bool = True


class CharBertEncoder(nn.Module):
    def __init__(self, cfg: CharBertConfig):
        super().__init__()
        self.cfg = cfg
        self.e_char = nn.Embedding(cfg.n_char_ids, cfg.d_model)
        self.e_bnd = nn.Embedding(cfg.n_boundary, cfg.d_model)
        self.e_dia = nn.Embedding(cfg.n_dia, cfg.d_model)
        self.e_punct = nn.Embedding(cfg.n_punct, cfg.d_model)
        self.e_region = nn.Embedding(cfg.n_region, cfg.d_model) if cfg.n_region > 0 else None
        self.e_century = nn.Embedding(cfg.n_century, cfg.d_model) if cfg.n_century > 0 else None
        rope = RoPE(cfg.d_model // cfg.n_heads)
        blocks = []
        for i in range(cfg.depth):
            win = 0 if i % 4 == 3 else cfg.char_window     # 3 local : 1 global
            blocks.append(Block(cfg.d_model, cfg.n_heads, rope, window=win, qk_norm=cfg.qk_norm))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = RMSNorm(cfg.d_model)
        self.head_char = nn.Linear(cfg.d_model, cfg.n_char_ids, bias=False)
        self.head_bnd = nn.Linear(cfg.d_model, 3, bias=False)
        self.head_dia = nn.Linear(cfg.d_model, 48, bias=False)
        self.head_cap = nn.Linear(cfg.d_model, 2, bias=False)
        self.head_punct = nn.Linear(cfg.d_model, 6, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, batch):
        cfg = self.cfg
        ids = batch["input_ids"]                            # (B,T)
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        seg = batch["seg_id"]                               # doc id per position (packing)

        x = (self.e_char(ids) + self.e_bnd(batch["boundary"]) + self.e_dia(batch["dia"])
             + self.e_punct(batch["punct"]))
        if self.e_region is not None:
            x = x + self.e_region(batch["region"])          # (B,T) per-position, same id across a doc's span
        if self.e_century is not None:
            x = x + self.e_century(batch["century"])         # (B,T) per-position, same id across a doc's span

        if cfg.attn_impl == "flex":
            char_mask = build_block_mask(seg, cfg.char_window, ids.device)
            glob_mask = build_block_mask(seg, 0, ids.device)
        else:
            char_mask = build_attn_mask(seg, cfg.char_window, ids.device, x.dtype)
            glob_mask = build_attn_mask(seg, 0, ids.device, x.dtype)

        for blk in self.blocks:
            m = glob_mask if blk.window == 0 else char_mask
            x = blk(x, pos, m)

        x = self.norm_out(x)
        return dict(
            char=self.head_char(x),
            boundary=self.head_bnd(x),
            dia=self.head_dia(x),
            cap=self.head_cap(x),
            punct=self.head_punct(x),
        )


def num_params(m):
    return sum(p.numel() for p in m.parameters())
