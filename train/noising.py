"""Elastic masked-diffusion noising (plan §4).

Given a packed batch of char sequences with aligned boundary planes, produce:
  input_ids   corrupted char ids (MASK where noised)
  labels      target char ids at supervised positions, -100 elsewhere
  loss_w      per-position loss weight (MDLM 1/t reweighting, averaged over the batch)
  keep_bnd_mask  per-POSITION bool: is the boundary channel known as input here
  keep_dia_mask  per-POSITION bool: is the diacritic channel known as input here

Noise patterns, chosen per sequence by a mixture (weights configurable), rate t ~ [0.05, 0.95]:
  span      contiguous char run, boundary-agnostic (geometric length) -> the realistic damage
            shape: a broken stone or torn papyrus doesn't respect word edges.
  word      whole word(s) via boundary plane, frequency-weighted so stopwords don't dominate.
  halfword  a PARTIAL word, anchored to its beginning, middle, or end (weighted toward the end:
            Greek is heavily suffixal, so word-final spans train ending/inflection restoration
            directly; word-initial spans cover augments/prefixes; medial spans cover internal
            damage, e.g. a lost dichronon).
  elastic   a word span replaced by a VARIABLE number of MASK slots (true length L ->
            M in [L, ceil(1.3L)+2]); targets are the L chars followed by (M-L) BLANK(∅) tokens.
            Teaches variable-length infilling + a gap-length signal, with no decoder.
  iid       independent per-char masking at rate t.

The elastic pattern changes sequence length, so noising returns a NEW packed batch
(lengths differ from the input); the collate step must therefore run noising before
building attention/segment masks. Everything here is pure PyTorch, no CUDA specifics,
so it is testable on CPU.

Channel availability (boundary/diacritics) is noised INDEPENDENTLY of char masking, per
CHARACTER POSITION (not per whole sequence): a mixture of fully-known, fully-unknown, and
genuinely patchy (rate ~Uniform(0,1) applied per position) — modeling real epigraphic/
papyrological cases where SOME word-breaks are legible and others are damaged, or SOME
accents survive and others don't, within the same text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# reserved id layout (must match model embedding table)
#   0..V-1        char ids (V = alphabet size, currently 24)
#   MASK, BLANK, PAD are appended after the alphabet
@dataclass
class NoiseConfig:
    vocab: int = 24
    mask_id: int = 24
    blank_id: int = 25          # ∅ : "gap ends / no char here"
    pad_id: int = 26
    # mixture weights over patterns (need not sum to 1; normalized). pat index: 0=span,
    # 1=word, 2=elastic, 3=iid, 4=halfword, 5=substitute.
    w_span: float = 0.15
    w_word: float = 0.20
    w_elastic: float = 0.15
    w_iid: float = 0.15
    w_halfword: float = 0.20
    w_substitute: float = 0.15  # DENOISING: scattered chars replaced by a WRONG letter (no
                                # mask signal) — real repaired-OCR/scribal text has substitution
                                # errors, not just gaps; the model must catch+correct these too.
    # rate distribution: t ~ Beta(a,b) clipped to [t_min, t_max]
    t_min: float = 0.05
    t_max: float = 0.95
    beta_a: float = 2.0
    beta_b: float = 4.0         # mean ~0.33, mass in the 0.15-0.45 encoder-quality band
    span_mean: float = 3.5      # geometric mean length for span/word-ish spans
    span_max: int = 20
    elastic_pad_frac: float = 0.3   # M = L + Bernoulli-ish extra up to ceil(elastic_pad_frac*L)+2
    elastic_extra_min: int = 2
    # halfword anchor mixture: weighted toward word-END (Greek inflection is suffixal)
    halfword_end_p: float = 0.5
    halfword_begin_p: float = 0.3
    halfword_mid_p: float = 0.2
    # per-position channel-availability mixture: P(fully known) + P(fully unknown) + the
    # remainder is genuinely patchy (per-position Bernoulli at a rate ~ Uniform(0,1))
    p_bnd_full: float = 0.5
    p_bnd_none: float = 0.3
    p_dia_full: float = 0.3
    p_dia_none: float = 0.5
    p_punct_full: float = 0.3
    p_punct_none: float = 0.5
    # per-DOCUMENT (not per-position) metadata dropout: real region/century values are only
    # sometimes known for a real fragment (unprovenanced papyri, no surviving date), so the
    # model must be trained under both conditions to be usable in either at inference. Default
    # 0.0 preserves every existing caller's exact behavior (metadata conditioning off unless
    # a caller explicitly sets these).
    p_region_none: float = 0.0
    p_century_none: float = 0.0


def _sample_t(cfg, n, g):
    a = torch.full((n,), cfg.beta_a)
    b = torch.full((n,), cfg.beta_b)
    t = torch.distributions.Beta(a, b).sample()
    return t.clamp(cfg.t_min, cfg.t_max)


def _mdlm_weight(t):
    # MDLM continuous-time weight ~ 1/t (clamped); normalized later per batch
    return (1.0 / t.clamp_min(0.05))


def _sample_keep_mask(n, g, p_full, p_none):
    """Per-position channel-availability mask (True = known/kept at this position). Mixture of
    fully-known, fully-unknown, and patchy (rate ~ Uniform(0,1), independent per position) —
    the patchy case is what lets one sequence have SOME boundaries/accents known and others not."""
    u = torch.rand(1, generator=g).item()
    if u < p_full:
        return torch.ones(n, dtype=torch.bool)
    if u < p_full + p_none:
        return torch.zeros(n, dtype=torch.bool)
    rate = torch.rand(1, generator=g).item()
    return torch.rand(n, generator=g) < rate


def _pick_halfword_span(s, e, cfg, g):
    """s,e: char range [s,e) of one word. Returns a sub-span anchored to the word's beginning,
    middle, or end (weighted toward the end — see module docstring)."""
    L = e - s
    if L <= 1:
        return s, e
    u = torch.rand(1, generator=g).item()
    anchor = "end" if u < cfg.halfword_end_p else (
             "begin" if u < cfg.halfword_end_p + cfg.halfword_begin_p else "mid")
    l = int(torch.randint(1, L + 1, (1,), generator=g).item())
    if anchor == "begin":
        return s, s + l
    if anchor == "end":
        return e - l, e
    if L <= 2:                       # too short for a non-edge-touching middle span
        return s, e
    l = max(1, min(l, L - 2))
    off = int(torch.randint(1, L - l, (1,), generator=g).item()) if L - l > 1 else 1
    return s + off, s + off + l


def noise_sequence(chars, boundary, cfg: NoiseConfig, g: torch.Generator, is_real_lacuna=None):
    """Corrupt ONE sequence (1D LongTensor chars, 1D boundary). Returns dict of 1D tensors.

    is_real_lacuna: optional bool tensor, same length as chars. Marks positions where the
    true content is GENUINELY unknown (a real '-'/'...' run from the edition itself --
    see insc/data/iphi.py's/papyri.py's text_to_full_planes()), already fed as mask_id in
    `chars`. When given, these positions are never eligible to be chosen as an ADDITIONAL
    synthetic-masking target (there's nothing there to mask further, and `chars` at those
    positions is mask_id, not a real letter -- treating it as one would produce a nonsense
    label) and their label always stays -100. None (default) preserves every existing
    caller's behavior exactly -- this is purely additive."""
    n = chars.numel()
    device = chars.device
    t = _sample_t(cfg, 1, g).item()

    # choose pattern: 0=span, 1=word, 2=elastic, 3=iid, 4=halfword, 5=substitute
    weights = torch.tensor([cfg.w_span, cfg.w_word, cfg.w_elastic, cfg.w_iid, cfg.w_halfword,
                           cfg.w_substitute], dtype=torch.float)
    pat = torch.multinomial(weights, 1, generator=g).item()

    inp = chars.clone()
    lab = torch.full((n,), -100, dtype=torch.long, device=device)

    def mask_positions(pos):
        inp[pos] = cfg.mask_id
        lab[pos] = chars[pos]

    def substitute_positions(pos):
        # DENOISING: replace with a WRONG letter (never MASK) — no signal that it's corrupted.
        wrong = (chars[pos] + torch.randint(1, cfg.vocab, (len(pos),), generator=g,
                                            device=device)) % cfg.vocab
        inp[pos] = wrong
        lab[pos] = chars[pos]

    if pat == 3:  # iid (mask)
        m = torch.rand(n, generator=g, device=device) < t
        if is_real_lacuna is not None:
            m = m & ~is_real_lacuna
        if m.any():
            mask_positions(m.nonzero(as_tuple=True)[0])
        return _finish(inp, lab, chars, boundary, t, cfg, g)

    if pat == 5:  # substitute (denoise) — same scattered selection as iid, no mask token
        m = torch.rand(n, generator=g, device=device) < t
        if is_real_lacuna is not None:
            m = m & ~is_real_lacuna
        if m.any():
            substitute_positions(m.nonzero(as_tuple=True)[0])
        return _finish(inp, lab, chars, boundary, t, cfg, g)

    # span/word/elastic/halfword all pick target spans until ~t fraction of chars covered
    word_ends = (boundary >= 1).nonzero(as_tuple=True)[0]
    # word start indices
    starts = torch.cat([torch.tensor([0], device=device), word_ends[:-1] + 1])
    words = list(zip(starts.tolist(), (word_ends + 1).tolist())) if word_ends.numel() else [(0, n)]

    budget = int(t * n)
    covered = 0
    chosen_spans = []
    tries = 0
    # pre-mark real-lacuna positions as "used" -- span selection below already rejects any
    # candidate overlapping a used position, so this alone keeps every chosen span entirely
    # within genuinely-known text without any extra branching in the selection loop.
    used = (is_real_lacuna.clone() if is_real_lacuna is not None
            else torch.zeros(n, dtype=torch.bool, device=device))
    while covered < budget and tries < 4 * len(words) + 8:
        tries += 1
        if pat == 0:  # span: random contiguous run, boundary-agnostic
            L = min(int(torch.distributions.Geometric(1.0 / cfg.span_mean).sample().item()) + 1,
                    cfg.span_max)
            s = int(torch.randint(0, max(n - L, 1), (1,), generator=g).item())
            e = min(s + L, n)
        elif pat == 4:  # halfword: partial word, anchored begin/middle/end
            wi = int(torch.randint(0, len(words), (1,), generator=g).item())
            ws, we = words[wi]
            s, e = _pick_halfword_span(ws, we, cfg, g)
        else:         # word / elastic: pick a whole word
            wi = int(torch.randint(0, len(words), (1,), generator=g).item())
            s, e = words[wi]
        if s >= e or used[s:e].any():
            continue
        used[s:e] = True
        chosen_spans.append((s, e))
        covered += e - s

    if pat in (0, 1, 4):  # span / word / halfword: fixed-length mask
        for s, e in chosen_spans:
            mask_positions(torch.arange(s, e, device=device))
        return _finish(inp, lab, chars, boundary, t, cfg, g)

    # pat == 2 elastic: rebuild the sequence with variable mask runs. chosen_spans is already
    # guaranteed lacuna-free (see `used` above); every other position -- including any real
    # lacuna -- is copied through unchanged with label -100 by _elastic_rebuild's own walk,
    # so no further is_real_lacuna handling is needed once the sequence is rebuilt.
    return _elastic_rebuild(chars, boundary, chosen_spans, t, cfg, g)


def _elastic_rebuild(chars, boundary, spans, t, cfg, g):
    device = chars.device
    n = chars.numel()
    spanset = sorted(spans)
    out_in, out_lab, out_bnd = [], [], []
    i = 0
    span_i = 0
    while i < n:
        if span_i < len(spanset) and i == spanset[span_i][0]:
            s, e = spanset[span_i]; span_i += 1
            L = e - s
            extra = cfg.elastic_extra_min + int(
                torch.randint(0, int(cfg.elastic_pad_frac * L) + 1, (1,), generator=g).item())
            M = L + extra
            out_in.append(torch.full((M,), cfg.mask_id, dtype=torch.long, device=device))
            lab = torch.full((M,), -100, dtype=torch.long, device=device)
            lab[:L] = chars[s:e]
            lab[L:] = cfg.blank_id          # predict ∅ for the surplus slots
            out_lab.append(lab)
            # boundary target for the gap: internal until the real word-end, unknown handled by model
            b = torch.zeros(M, dtype=torch.uint8, device=device)
            b[L - 1] = boundary[e - 1]
            out_bnd.append(b)
            i = e
        else:
            out_in.append(chars[i:i + 1])
            out_lab.append(torch.tensor([-100], device=device))
            out_bnd.append(boundary[i:i + 1])
            i += 1
    inp = torch.cat(out_in)
    lab = torch.cat(out_lab)
    bnd = torch.cat(out_bnd)
    return _finish(inp, lab, chars, bnd, t, cfg, g, rebuilt=True)


def _finish(inp, lab, orig_chars, boundary, t, cfg, g, rebuilt=False):
    n = inp.numel()
    keep_bnd_mask = _sample_keep_mask(n, g, cfg.p_bnd_full, cfg.p_bnd_none)
    keep_dia_mask = _sample_keep_mask(n, g, cfg.p_dia_full, cfg.p_dia_none)
    keep_punct_mask = _sample_keep_mask(n, g, cfg.p_punct_full, cfg.p_punct_none)
    w = _mdlm_weight(torch.tensor(t))
    return dict(input_ids=inp, labels=lab, boundary=boundary,
                loss_w=torch.full_like(inp, float(w), dtype=torch.float),
                keep_bnd_mask=keep_bnd_mask, keep_dia_mask=keep_dia_mask,
                keep_punct_mask=keep_punct_mask, t=t, rebuilt=rebuilt)
