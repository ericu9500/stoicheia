"""TaggerModel: Stoicheia encoder + word pooling + factored XPOS / edit-script / UPOS heads."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tagger.backbone import CharBertWithHidden


@dataclass
class TaggerConfig:
    pool: str = "mean"          # "mean" | "last"
    head_dropout: float = 0.1
    w_xpos: float = 1.0         # factored 9-position heads
    w_flat: float = 0.0         # flat full-tag head (attested tags); 0 disables
    w_script: float = 1.0
    w_upos: float = 0.2
    use_cap: bool = False       # inject a zero-init capitalization embedding (fine-tune only)
    scalar_mix: bool = False    # ELMo-style learned mix over all block outputs


def pool_words(hidden, word_id, W, mode="mean"):
    """hidden (B,T,D), word_id (B,T) in [-1,W) -> (B,W,D)."""
    B, T, D = hidden.shape
    flat = hidden.reshape(B * T, D)
    wid = word_id.reshape(B * T)
    valid = wid >= 0
    off = (torch.arange(B, device=hidden.device) * W).repeat_interleave(T)
    idx = (wid + off)[valid]
    out = hidden.new_zeros(B * W, D)
    if mode == "mean":
        out.index_add_(0, idx, flat[valid])
        cnt = hidden.new_zeros(B * W).index_add_(
            0, idx, torch.ones_like(idx, dtype=hidden.dtype))
        out = out / cnt.clamp(min=1).unsqueeze(-1)
    elif mode == "last":
        out.index_copy_(0, idx, flat[valid])   # spans are contiguous: last write = last char
    else:
        raise ValueError(mode)
    return out.reshape(B, W, D)


class TaggerModel(nn.Module):
    def __init__(self, encoder: CharBertWithHidden, vocab, tcfg: TaggerConfig, W=384):
        super().__init__()
        self.encoder = encoder
        self.tcfg = tcfg
        self.W = W
        d = encoder.cfg.d_model
        if tcfg.use_cap:
            emb = nn.Embedding(2, d)
            nn.init.zeros_(emb.weight)
            encoder.cap_emb = emb          # picked up by CharBertWithHidden.forward
        self.dropout = nn.Dropout(tcfg.head_dropout)
        self.xpos_heads = nn.ModuleList(
            [nn.Linear(d, len(a), bias=False) for a in vocab.xpos_alpha])
        self.head_flat = (nn.Linear(d, len(vocab.tags), bias=False)
                          if tcfg.w_flat > 0 else None)
        self.head_script = nn.Linear(d, vocab.n_scripts, bias=False)
        self.head_upos = nn.Linear(d, len(vocab.upos), bias=False)
        if tcfg.scalar_mix:
            encoder.return_layers = True
            # blocks + final normed hidden; zero-init = uniform mix at start
            self.mix_w = nn.Parameter(torch.zeros(len(encoder.blocks) + 1))
        for m in [*self.xpos_heads, self.head_script, self.head_upos,
                  *( [self.head_flat] if self.head_flat is not None else [] )]:
            nn.init.normal_(m.weight, std=0.02)
        # pretraining output heads take no part in the tagging loss; freeze them so DDP
        # doesn't trip on parameters that never receive gradients. CharBertWithHidden always
        # has these; a HF backbone (tagger.hf_backbone.HFBackboneWithHidden) has none of them,
        # so this is a no-op there -- getattr guards keep TaggerModel encoder-agnostic.
        for name in ("head_char", "head_bnd", "head_dia", "head_cap", "head_punct"):
            m = getattr(encoder, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, batch):
        out = self.encoder(batch)
        if self.tcfg.scalar_mix:
            # pooling is linear, so pool per layer then mix (much smaller than mixing (B,T,D))
            pooled = torch.stack(
                [pool_words(h, batch["word_id"], self.W, self.tcfg.pool)
                 for h in [*out["layers"], out["hidden"]]])      # (L+1,B,W,D)
            mix = torch.softmax(self.mix_w, 0)
            w = torch.einsum("l,lbwd->bwd", mix.to(pooled.dtype), pooled)
        else:
            w = pool_words(out["hidden"], batch["word_id"], self.W, self.tcfg.pool)
        w = self.dropout(w)
        r = dict(xpos=[hd(w) for hd in self.xpos_heads],
                 script=self.head_script(w),
                 upos=self.head_upos(w))
        if self.head_flat is not None:
            r["flat"] = self.head_flat(w)
        return r

    @staticmethod
    def _ce(logits, target):
        """CE that stays finite (and keeps the head in the DDP graph) when a batch has
        no valid labels for this task — e.g. silver lemma-distillation batches."""
        if bool((target != -100).any()):
            return F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100)
        return logits.sum() * 0.0

    def loss(self, out, batch):
        t = self.tcfg
        xl = [self._ce(lg, batch["y_xpos"][:, :, p]) for p, lg in enumerate(out["xpos"])]
        l_x = torch.stack(xl).mean()
        l_s = self._ce(out["script"], batch["y_script"])
        l_u = self._ce(out["upos"], batch["y_upos"])
        loss = t.w_xpos * l_x + t.w_script * l_s + t.w_upos * l_u
        logs = dict(l_xpos=round(l_x.item(), 4), l_script=round(l_s.item(), 4),
                    l_upos=round(l_u.item(), 4))
        if self.head_flat is not None:
            l_f = self._ce(out["flat"], batch["y_tag"])
            loss = loss + t.w_flat * l_f
            logs["l_flat"] = round(l_f.item(), 4)
        return loss, logs
