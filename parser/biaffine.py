"""Deep biaffine dependency parser (Dozat & Manning 2017) over a frozen Stoicheia
encoder, combined via an ELMo-style learned scalar mix of all layers (same recipe the
tagger used for XPOS/lemma: scalar_mix + light head, here applied to arc/label MLPs
instead of tag heads).

Backbones are FROZEN (only the scalar-mix weights + biaffine head train) — this keeps a fair,
fast 3-way ablation (char / lemma / fused) without re-touching either finished pretraining run.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMix(nn.Module):
    """Learned softmax-weighted sum over N layer outputs (+ a global scale), ELMo-style."""
    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, layers):           # layers: list of (B,T,D), len n_layers
        w = torch.softmax(self.w, 0)
        mixed = sum(wi * h for wi, h in zip(w, layers))
        return self.gamma * mixed


def pool_words(hidden, word_id, W, mode="mean"):
    B, T, D = hidden.shape
    flat = hidden.reshape(B * T, D)
    wid = word_id.reshape(B * T)
    valid = wid >= 0
    off = (torch.arange(B, device=hidden.device) * W).repeat_interleave(T)
    idx = (wid + off)[valid]
    out = hidden.new_zeros(B * W, D)
    out.index_add_(0, idx, flat[valid])
    cnt = hidden.new_zeros(B * W).index_add_(0, idx, torch.ones_like(idx, dtype=hidden.dtype))
    out = out / cnt.clamp(min=1).unsqueeze(-1)
    return out.reshape(B, W, D)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.33):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.lin(x)))


class Biaffine(nn.Module):
    """s(x,y) = [x;1]^T W [y;1] (per output channel). x:(B,Lx,Di) y:(B,Ly,Di) -> (B,n_out,Lx,Ly)."""
    def __init__(self, d_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.W = nn.Parameter(torch.zeros(n_out, d_in + int(bias_x), d_in + int(bias_y)))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], -1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.W, y)
        return s.squeeze(1) if s.shape[1] == 1 else s


@dataclass
class ParserConfig:
    d_arc: int = 500
    d_rel: int = 150
    dropout: float = 0.33
    n_labels: int = 40


class BiaffineHead(nn.Module):
    """Arc + label scoring over word vectors (B,W,D). A learnable ROOT vector is the head
    candidate for column 0 (D&M's pseudo-root). Arc loss / label loss are standard CE."""
    def __init__(self, d_in, cfg: ParserConfig):
        super().__init__()
        self.cfg = cfg
        self.root = nn.Parameter(torch.zeros(d_in))
        nn.init.normal_(self.root, std=0.02)
        self.arc_dep = MLP(d_in, cfg.d_arc, cfg.dropout)
        self.arc_head = MLP(d_in, cfg.d_arc, cfg.dropout)
        self.rel_dep = MLP(d_in, cfg.d_rel, cfg.dropout)
        self.rel_head = MLP(d_in, cfg.d_rel, cfg.dropout)
        self.arc_biaf = Biaffine(cfg.d_arc, n_out=1, bias_x=True, bias_y=False)
        self.rel_biaf = Biaffine(cfg.d_rel, n_out=cfg.n_labels, bias_x=True, bias_y=True)

    def forward(self, w, word_mask):
        """w: (B,W,D) word vectors. word_mask: (B,W) bool, True at real words.
        Returns arc_scores (B,W,W+1) [col0=root], rel_scores (B,W,W+1,n_labels)."""
        B, W, D = w.shape
        root = self.root.view(1, 1, D).expand(B, 1, D)
        heads_in = torch.cat([root, w], 1)                       # (B,W+1,D): col0=root
        h_dep_arc = self.arc_dep(w)                               # (B,W,d_arc)
        h_head_arc = self.arc_head(heads_in)                      # (B,W+1,d_arc)
        arc_scores = self.arc_biaf(h_dep_arc, h_head_arc)          # (B,W,W+1)
        # mask: dependent i cannot pick itself as head (col i+1), and padded cols get -inf
        pad_head = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=w.device), word_mask], 1)
        arc_scores = arc_scores.masked_fill(~pad_head[:, None, :], float("-inf"))
        self_idx = torch.arange(W, device=w.device)
        arc_scores[:, self_idx, self_idx + 1] = float("-inf")

        h_dep_rel = self.rel_dep(w)                                # (B,W,d_rel)
        h_head_rel = self.rel_head(heads_in)                       # (B,W+1,d_rel)
        rel_scores = self.rel_biaf(h_dep_rel, h_head_rel)           # (B,n_labels,W,W+1)
        rel_scores = rel_scores.permute(0, 2, 3, 1)                 # (B,W,W+1,n_labels)
        return arc_scores, rel_scores

    def loss(self, arc_scores, rel_scores, heads, labels, word_mask):
        """heads: (B,W) gold head col index (0=root, else 1..W); labels: (B,W) gold label id;
        both -100 where padded/not-a-word."""
        m = heads != -100
        arc_loss = F.cross_entropy(arc_scores[m], heads[m])
        B, W = heads.shape
        bi = torch.arange(B, device=heads.device)[:, None].expand(B, W)
        wi = torch.arange(W, device=heads.device)[None, :].expand(B, W)
        gold_head = heads.clamp(min=0)
        sel = rel_scores[bi, wi, gold_head]                        # (B,W,n_labels)
        rel_loss = F.cross_entropy(sel[m], labels[m])
        return arc_loss + rel_loss, dict(arc=round(arc_loss.item(), 4), rel=round(rel_loss.item(), 4))

    @torch.no_grad()
    def decode(self, arc_scores, rel_scores, word_mask):
        """Greedy per-token argmax head (col 0=root) + label argmax at the chosen head.
        Not tree-constrained (no MST projection) — the official conll18 LAS/UAS scorer
        compares HEAD/DEPREL per token regardless of global tree-validity, so this is a
        correct, simple decode for that metric (the standard simplification vs. full MST)."""
        heads_out = arc_scores.argmax(-1)                          # (B,W) in [0..W], 0=root
        B, W = heads_out.shape
        bi = torch.arange(B, device=arc_scores.device)[:, None].expand(B, W)
        wi = torch.arange(W, device=arc_scores.device)[None, :].expand(B, W)
        labels_out = rel_scores[bi, wi, heads_out].argmax(-1)
        return heads_out.cpu(), labels_out.cpu()
