"""Masked-diffusion corruption for the token-level ablation arms (flat characters, BPE subwords).

Mirrors train/noising.py pattern for pattern, with the unit changed from "letter" to "token":
  span      contiguous token run, geometric length, word-edge agnostic
  word      whole word(s)  (a word = a "▁"-initial token and its continuation tokens)
  halfword  a partial word anchored to its beginning / middle / end
  elastic   a word replaced by M >= L mask slots; the target is the L tokens then (M-L) [BLANK]
  iid       independent per-token masking at rate t
  substitute a wrong token with no mask marker (denoising)
Rate t ~ Beta(2,4) clipped to [0.05, 0.95]; MDLM 1/t loss weight — identical to the plane model.

Channel dropout: the plane model hides its diacritic channel per position (fully / not at all /
patchy). The flat-character arm gets the analogous corruption: at "dropped" positions the input
character is replaced by its accentless form and the original is supervised by the auxiliary loss
(weight lam, as in train/loss.py). BPE tokens cannot be stripped in place, so the subword arm has
no channel dropout (documented asymmetry; it only removes a task from that arm).
Everything is numpy on CPU (runs in DataLoader workers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD, MASK, BLANK, UNK = 0, 1, 2, 3


@dataclass
class TokNoiseConfig:
    w_span: float = 0.15
    w_word: float = 0.20
    w_elastic: float = 0.15
    w_iid: float = 0.15
    w_halfword: float = 0.20
    w_substitute: float = 0.15
    t_min: float = 0.05
    t_max: float = 0.95
    beta_a: float = 2.0
    beta_b: float = 4.0
    span_mean: float = 3.5
    span_max: int = 20
    elastic_pad_frac: float = 0.3
    elastic_extra_min: int = 2
    halfword_end_p: float = 0.5
    halfword_begin_p: float = 0.3
    p_dia_full: float = 0.3
    p_dia_none: float = 0.5


def _keep_mask(n, rng, p_full, p_none):
    u = rng.random()
    if u < p_full:
        return np.ones(n, bool)
    if u < p_full + p_none:
        return np.zeros(n, bool)
    return rng.random(n) < rng.random()


def _halfword(s, e, cfg, rng):
    L = e - s
    if L <= 1:
        return s, e
    u = rng.random()
    l = int(rng.integers(1, L + 1))
    if u < cfg.halfword_end_p:
        return e - l, e
    if u < cfg.halfword_end_p + cfg.halfword_begin_p:
        return s, s + l
    if L <= 2:
        return s, e
    l = max(1, min(l, L - 2))
    off = int(rng.integers(1, L - l)) if L - l > 1 else 1
    return s + off, s + off + l


def noise_tokens(ids: np.ndarray, word_start: np.ndarray, cfg: TokNoiseConfig, rng,
                 sub_ids: np.ndarray, strip_map: np.ndarray | None = None,
                 protect: np.ndarray | None = None):
    """ids: int64 token ids of one record; word_start: bool per token; protect: optional bool per
    token marking REAL lacunae (input MASK, never supervised, never chosen as a synthetic target —
    the analogue of train/noising.py's is_real_lacuna).
    Returns dict(input_ids, labels, loss_w (scalar), aux_labels) as int64 arrays."""
    n = len(ids)
    if protect is None:
        protect = np.zeros(n, bool)
    ids = np.where(protect, MASK, ids)
    t = float(np.clip(rng.beta(cfg.beta_a, cfg.beta_b), cfg.t_min, cfg.t_max))
    w = np.array([cfg.w_span, cfg.w_word, cfg.w_elastic, cfg.w_iid, cfg.w_halfword,
                  cfg.w_substitute], float)
    pat = int(rng.choice(6, p=w / w.sum()))
    inp = ids.copy()
    lab = np.full(n, -100, np.int64)

    if pat in (3, 5):
        m = (rng.random(n) < t) & ~protect
        if pat == 3:
            inp[m] = MASK
        else:
            k = int(m.sum())
            if k:
                wrong = sub_ids[rng.integers(0, len(sub_ids), k)]
                same = wrong == ids[m]
                wrong[same] = sub_ids[(np.searchsorted(sub_ids, wrong[same]) + 1) % len(sub_ids)]
                inp[m] = wrong
        lab[m] = ids[m]
        return _finish(inp, lab, np.where(protect, PAD, ids), t, cfg, rng, strip_map)

    ws = np.flatnonzero(word_start)
    if len(ws) == 0 or ws[0] != 0:
        ws = np.concatenate([[0], ws])
    words = list(zip(ws.tolist(), np.concatenate([ws[1:], [n]]).tolist()))
    budget = int(t * n)
    covered, tries, spans = 0, 0, []
    used = protect.copy()
    while covered < budget and tries < 4 * len(words) + 8:
        tries += 1
        if pat == 0:
            L = min(int(rng.geometric(1.0 / cfg.span_mean)), cfg.span_max)
            s = int(rng.integers(0, max(n - L, 1))); e = min(s + L, n)
        elif pat == 4:
            s, e = _halfword(*words[int(rng.integers(0, len(words)))], cfg, rng)
        else:
            s, e = words[int(rng.integers(0, len(words)))]
        if s >= e or used[s:e].any():
            continue
        used[s:e] = True
        spans.append((s, e)); covered += e - s

    if pat in (0, 1, 4):
        for s, e in spans:
            inp[s:e] = MASK; lab[s:e] = ids[s:e]
        return _finish(inp, lab, np.where(protect, PAD, ids), t, cfg, rng, strip_map)

    # elastic rebuild
    spans.sort()
    out_in, out_lab, out_orig = [], [], []
    i = si = 0
    while i < n:
        if si < len(spans) and i == spans[si][0]:
            s, e = spans[si]; si += 1
            L = e - s
            M = L + cfg.elastic_extra_min + int(rng.integers(0, int(cfg.elastic_pad_frac * L) + 1))
            out_in.append(np.full(M, MASK, np.int64))
            l = np.full(M, BLANK, np.int64); l[:L] = ids[s:e]
            out_lab.append(l)
            out_orig.append(np.full(M, PAD, np.int64))     # no channel dropout inside a gap
            i = e
        else:
            j = spans[si][0] if si < len(spans) else n
            out_in.append(ids[i:j]); out_lab.append(np.full(j - i, -100, np.int64))
            out_orig.append(np.where(protect[i:j], PAD, ids[i:j])); i = j
    inp = np.concatenate(out_in); lab = np.concatenate(out_lab)
    return _finish(inp, lab, np.concatenate(out_orig), t, cfg, rng, strip_map)


def _finish(inp, lab, orig, t, cfg, rng, strip_map):
    aux = np.full(len(inp), -100, np.int64)
    if strip_map is not None:
        visible = inp == orig                                # unmasked, unsubstituted
        keep = _keep_mask(len(inp), rng, cfg.p_dia_full, cfg.p_dia_none)
        stripped = strip_map[orig]
        drop = visible & ~keep & (stripped != orig)
        inp = np.where(drop, stripped, inp)
        aux[drop] = orig[drop]
    return dict(input_ids=inp, labels=lab, aux_labels=aux, loss_w=1.0 / max(t, 0.05))
