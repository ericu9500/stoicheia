"""WSD (warmup-stable-decay / trapezoidal) LR schedule (plan §6).

Flat peak in the middle so we can branch/anneal at any point (seed-and-soup, epoch probe).
Decay phase = the gold-only anneal window. Returns a multiplier in [0,1] of peak LR.
"""
from __future__ import annotations

import math


def wsd(step, total, warmup_frac=0.04, decay_frac=0.2, min_frac=0.02):
    warm = int(total * warmup_frac)
    decay_start = int(total * (1 - decay_frac))
    if step < warm:
        return step / max(warm, 1)
    if step < decay_start:
        return 1.0
    # cosine decay to min_frac over the decay window
    t = (step - decay_start) / max(total - decay_start, 1)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


def in_anneal(step, total, decay_frac=0.2):
    """True once we enter the decay window (loader switches to gold-only)."""
    return step >= int(total * (1 - decay_frac))


def wsd_dyn(step, total, anneal_start, warmup_frac=0.04, decay_frac=0.2, min_frac=0.02):
    """WSD with a dynamic anneal point. anneal_start <= the planned decay start; the decay
    window keeps its planned LENGTH (decay_frac * total), so an early anneal finishes the
    run early rather than stretching the decay. With anneal_start == planned start this is
    identical to wsd()."""
    warm = int(total * warmup_frac)
    decay_len = total - int(total * (1 - decay_frac))
    if step < warm:
        return step / max(warm, 1)
    if step < anneal_start:
        return 1.0
    t = min((step - anneal_start) / max(decay_len, 1), 1.0)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))
