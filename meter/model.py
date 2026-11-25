"""MeterModel: Stoicheia encoder + per-letter macron and scansion heads."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from meter.backbone import CharBertWithHidden


@dataclass
class MeterConfig:
    head_dropout: float = 0.33
    w_mac: float = 1.0            # 0 disables the task (head still built, frozen out of loss)
    w_scan: float = 1.0
    use_cap: bool = True          # zero-init capitalization input embedding
    scalar_mix: bool = True       # ELMo-style learned mix over all block outputs
    mac_class_w: list = field(default_factory=lambda: [1.0, 1.0])   # long, short
    scan_class_w: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


class MeterModel(nn.Module):
    def __init__(self, encoder: CharBertWithHidden, mcfg: MeterConfig):
        super().__init__()
        self.encoder = encoder
        self.mcfg = mcfg
        d = encoder.cfg.d_model
        if mcfg.use_cap:
            emb = nn.Embedding(2, d)
            nn.init.zeros_(emb.weight)
            encoder.cap_emb = emb          # picked up by CharBertWithHidden.forward
        self.dropout = nn.Dropout(mcfg.head_dropout)
        self.head_mac = nn.Linear(d, 2, bias=False)
        self.head_scan = nn.Linear(d, 4, bias=False)
        if mcfg.scalar_mix:
            encoder.return_layers = True
            # blocks + final normed hidden; zero-init = uniform mix at start
            self.mix_w = nn.Parameter(torch.zeros(len(encoder.blocks) + 1))
        for m in (self.head_mac, self.head_scan):
            nn.init.normal_(m.weight, std=0.02)
        self.register_buffer("mac_w", torch.tensor(mcfg.mac_class_w, dtype=torch.float32))
        self.register_buffer("scan_w", torch.tensor(mcfg.scan_class_w, dtype=torch.float32))
        # pretraining output heads take no part in the loss; freeze them so DDP
        # doesn't trip on parameters that never receive gradients
        for m in (encoder.head_char, encoder.head_bnd, encoder.head_dia,
                  encoder.head_cap, encoder.head_punct):
            for p in m.parameters():
                p.requires_grad_(False)

    def forward(self, batch):
        out = self.encoder(batch)
        if self.mcfg.scalar_mix:
            h = torch.stack([*out["layers"], out["hidden"]])      # (L+1,B,T,D)
            mix = torch.softmax(self.mix_w, 0)
            h = torch.einsum("l,lbtd->btd", mix.to(h.dtype), h)
        else:
            h = out["hidden"]
        h = self.dropout(h)
        return dict(mac=self.head_mac(h), scan=self.head_scan(h))

    def _ce(self, logits, target, weight):
        """Class-weighted CE that stays finite (and keeps the head in the DDP graph)
        when a batch has no valid labels for this task."""
        if bool((target != -100).any()):
            return F.cross_entropy(logits.transpose(1, 2), target,
                                   weight=weight.to(logits.dtype), ignore_index=-100)
        return logits.sum() * 0.0

    def loss(self, out, batch):
        t = self.mcfg
        l_m = self._ce(out["mac"], batch["y_mac"], self.mac_w)
        l_s = self._ce(out["scan"], batch["y_scan"], self.scan_w)
        loss = t.w_mac * l_m + t.w_scan * l_s
        return loss, dict(l_mac=round(l_m.item(), 4), l_scan=round(l_s.item(), 4))
