"""Token-level encoder for the representation ablation, and its adapter into the plane harnesses.

TokenEncoder uses exactly the plane model's transformer (model/layers.py: pre-norm RoPE blocks,
GeGLU, QK-norm, 3 local : 1 global attention); only the input embedding and output head differ.

TokenAsPlanes lets a TokenEncoder checkpoint be fine-tuned and evaluated by the UNCHANGED plane
harnesses (tagger/parser, meter): it receives the harness's plane batch, renders every packed
document back to text (ablate/render.py), tokenizes it, runs the encoder, and returns per-LETTER
hidden states in the plane layout (a letter reads the hidden state of the token that contains it).
Downstream heads, pooling, packing and recipes are therefore identical across all arms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model.layers import Block, RMSNorm, RoPE, build_attn_mask, build_block_mask


@dataclass
class TokConfig:
    vocab: int
    d_model: int = 768
    n_heads: int = 12
    depth: int = 12
    window: int = 320
    attn_impl: str = "flex"
    qk_norm: bool = True


class TokenEncoder(nn.Module):
    def __init__(self, cfg: TokConfig):
        super().__init__()
        self.cfg = cfg
        self.e_tok = nn.Embedding(cfg.vocab, cfg.d_model)
        rope = RoPE(cfg.d_model // cfg.n_heads)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.n_heads, rope, window=0 if i % 4 == 3 else cfg.window,
                  qk_norm=cfg.qk_norm) for i in range(cfg.depth))
        self.norm_out = RMSNorm(cfg.d_model)
        self.head_tok = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def trunk(self, x, seg, collect=False):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        if self.cfg.attn_impl == "flex":
            loc = build_block_mask(seg, self.cfg.window, x.device)
            glob = build_block_mask(seg, 0, x.device)
        else:
            loc = build_attn_mask(seg, self.cfg.window, x.device, x.dtype)
            glob = build_attn_mask(seg, 0, x.device, x.dtype)
        layers = []
        for blk in self.blocks:
            x = blk(x, pos, glob if blk.window == 0 else loc)
            if collect:
                layers.append(x)
        return self.norm_out(x), layers

    def forward(self, batch):
        h, _ = self.trunk(self.e_tok(batch["input_ids"]), batch["seg_id"])
        return dict(tok=self.head_tok(h))


def load_token_encoder(ckpt_path, device, attn_impl="sdpa", random_init=False):
    sd = torch.load(os.path.expandvars(ckpt_path), map_location="cpu")
    c = sd["cfg"]
    tcfg = TokConfig(vocab=c["vocab"], d_model=c["d_model"], n_heads=c["d_model"] // 64,
                     depth=c["depth"], window=c["window"], attn_impl=attn_impl,
                     qk_norm=c.get("qk_norm", True))
    enc = TokenEncoder(tcfg)
    if not random_init:
        enc.load_state_dict(sd["model"])
    return enc.to(device), c


class _NoParams(nn.Module):
    def forward(self, x):
        return x


class TokenAsPlanes(nn.Module):
    """Plane-harness-compatible wrapper (see module docstring). Exposes the attributes the
    tagger/meter code reads: cfg.d_model, blocks, return_layers, cap_emb (set by the head),
    e_char/e_bnd/e_dia/e_punct, norm_out, head_* (param-group walks and head freezing)."""

    def __init__(self, enc: TokenEncoder, tokenizer_dir: str, max_tokens: int):
        super().__init__()
        from transformers import PreTrainedTokenizerFast
        self.enc = enc
        self.cfg = enc.cfg
        self.blocks = enc.blocks
        self.norm_out = enc.norm_out
        self.e_char = enc.e_tok
        self.e_bnd = _NoParams(); self.e_dia = _NoParams(); self.e_punct = _NoParams()
        self.head_char = enc.head_tok
        self.head_bnd = _NoParams(); self.head_dia = _NoParams()
        self.head_cap = _NoParams(); self.head_punct = _NoParams()
        self.return_layers = False
        self.tok = PreTrainedTokenizerFast.from_pretrained(os.path.expandvars(tokenizer_dir))
        self.max_tokens = max_tokens

    def _encode_row(self, chars, bnd, dia, punct, cap, seg):
        """One packed plane row -> (token ids, token seg ids, letter->token index)."""
        from ablate.render import render
        T = len(chars)
        tok_ids, tok_seg = [], []
        l2t = np.zeros(T, np.int64)
        for s in np.unique(seg[seg > 0]):
            pos = np.flatnonzero(seg == s)
            letters = chars[pos]
            ok = letters < 24
            if not ok.all():            # masked letters render as nothing; harness inputs are full
                letters = np.where(ok, letters, 0)
            b = np.where(bnd[pos] >= 3, 0, bnd[pos])
            text, offs = render(letters, b, dia[pos], cap[pos], punct[pos])
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            char2tok = np.full(len(text) + 1, -1, np.int64)
            for ti, (a, e) in enumerate(enc["offset_mapping"]):
                char2tok[a:max(e, a + 1)] = ti
            # Metaspace offsets can exclude the "▁" slot; fill any gap from the right neighbour
            for k in range(len(char2tok) - 2, -1, -1):
                if char2tok[k] < 0:
                    char2tok[k] = char2tok[k + 1]
            base = len(tok_ids)
            l2t[pos] = base + char2tok[offs]
            tok_ids.extend(ids); tok_seg.extend([int(s)] * len(ids))
        return tok_ids, tok_seg, l2t

    def forward(self, batch):
        ids = batch["input_ids"]; B, T = ids.shape
        dev = ids.device
        get = lambda k, fill: (batch[k].cpu().numpy() if k in batch
                               else np.full((B, T), fill, np.int64))
        chars, bnd, dia, punct = get("input_ids", 0), get("boundary", 0), get("dia", 48), get("punct", 6)
        cap, seg = get("cap", 0), get("seg_id", 0)
        rows = [self._encode_row(chars[b], bnd[b], dia[b], punct[b], cap[b], seg[b]) for b in range(B)]
        L = max(max(len(r[0]) for r in rows), 1)
        if L > self.max_tokens:
            raise ValueError(f"token row {L} exceeds max_tokens={self.max_tokens}")
        tid = torch.zeros(B, L, dtype=torch.long); tseg = torch.zeros(B, L, dtype=torch.long)
        for b, (ti, ts, _) in enumerate(rows):
            tid[b, :len(ti)] = torch.tensor(ti); tseg[b, :len(ts)] = torch.tensor(ts)
        l2t = torch.tensor(np.stack([r[2] for r in rows]), device=dev)
        x = self.enc.e_tok(tid.to(dev))
        h, layers = self.enc.trunk(x, tseg.to(dev), collect=self.return_layers)
        gather = lambda z: torch.gather(z, 1, l2t[:, :, None].expand(B, T, z.shape[-1]))
        hid = gather(h)
        cap_emb = getattr(self, "cap_emb", None)
        if cap_emb is not None and "cap" in batch:
            # zero-init fine-tune channel (the plane model adds it at the input); here case is
            # already rendered into the text, so this only keeps the parameter in the graph
            hid = hid + cap_emb(batch["cap"])
        return dict(layers=[gather(z) for z in layers], hidden=hid)


def load_any_backbone(ckpt_path, device, attn_impl="sdpa"):
    """Dispatch used by tagger/meter backbones: token-arm checkpoints (cfg.arch == 'tok') come back
    wrapped in TokenAsPlanes; None means 'not a token checkpoint'."""
    random_init = str(ckpt_path).startswith("random:")
    path = str(ckpt_path).split(":", 1)[1] if random_init else str(ckpt_path)
    sd = torch.load(os.path.expandvars(path), map_location="cpu", weights_only=False)
    c = sd.get("cfg", {})
    if c.get("arch") != "tok":
        return None
    enc, c = load_token_encoder(path, device, attn_impl=attn_impl, random_init=random_init)
    return TokenAsPlanes(enc, c["tokenizer"], max_tokens=c.get("max_ft_tokens", 16384)).to(device), c
