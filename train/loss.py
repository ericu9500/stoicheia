"""Multi-head loss: char-diffusion loss dominates; boundary/diacritics/cap/punctuation are
down-weighted auxiliary heads, jointly trained."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _anchored_ce(logits, labels, weight, logs, name):
    """Cross-entropy on supervised positions; if none, return a ZERO term that still touches
    every head param so DDP never sees an 'unused parameter'."""
    mm = labels != -100
    if mm.any():
        l = F.cross_entropy(logits[mm], labels[mm])
        logs[name] = l.item()
        return weight * l
    return logits.sum() * 0.0                      # zero grad, but head is "used"


def compute_loss(out, batch, lam=0.1):
    logs = {}
    # char diffusion loss with MDLM 1/t reweighting
    cl = out["char"]
    lab = batch["labels"]
    m = lab != -100
    if m.any():
        ce = F.cross_entropy(cl[m], lab[m], reduction="none")
        w = batch["loss_w"][m]
        char_loss = (ce * w).sum() / w.sum()
    else:
        char_loss = cl.sum() * 0.0
    logs["char"] = char_loss.item()
    total = char_loss

    total = total + _anchored_ce(out["boundary"], batch["bnd_lab"], lam, logs, "bnd")
    total = total + _anchored_ce(out["dia"], batch["dia_lab"], lam, logs, "dia")
    total = total + _anchored_ce(out["cap"], batch["cap_lab"], lam, logs, "cap")
    total = total + _anchored_ce(out["punct"], batch["punct_lab"], lam, logs, "punct")

    logs["total"] = total.item()
    return total, logs
