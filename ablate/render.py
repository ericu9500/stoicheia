"""Render Stoicheia's aligned planes back to flat polytonic text, for the token-level ablation arms.

The flat-character and subword arms read exactly the same shard records, windows and tier
sampling as the plane model (train/data.py), and turn each record into a string here, so the
only thing that differs between arms is the input representation.

  text      NFC polytonic words (case, accents, breathings, iota subscript, diaeresis), a
            punctuation mark after the letter that carries it, single spaces between words.
  letter_offsets[i]  index in `text` of the character that renders letter i (always a starter
            codepoint, since every letter cluster is NFC-normalized separately).
"""
from __future__ import annotations

import unicodedata

import numpy as np

from data.normalize import ID2LETTER, _ACC_M, _BR_M, _MARK_CHARS, M_DIAER, M_IOTA, unpack_dia

UNK_DIA, UNK_PUNCT = 48, 6
PUNCT_CHARS = {1: ",", 2: "·", 3: ":", 4: ".", 5: ";"}

_CLUSTER_CACHE: dict = {}


def _cluster(letter: int, dia: int, cap: int, final_sigma: bool) -> str:
    key = (letter, dia, cap, final_sigma)
    s = _CLUSTER_CACHE.get(key)
    if s is not None:
        return s
    ch = ID2LETTER[letter]
    if final_sigma and ch == "σ" and not cap:
        ch = "ς"
    if cap:
        ch = ch.upper()
    s = ch
    if dia < 48:
        acc, br, io, dd = unpack_dia(int(dia))
        if br:
            s += _MARK_CHARS[_BR_M[br]]
        if dd:
            s += _MARK_CHARS[M_DIAER]
        if acc:
            s += _MARK_CHARS[_ACC_M[acc]]
        if io:
            s += _MARK_CHARS[M_IOTA]
    s = unicodedata.normalize("NFC", s)
    _CLUSTER_CACHE[key] = s
    return s


_TABLE = None      # (letter, dia 0..48, cap, final) -> rendered cluster; built once


def _table():
    global _TABLE
    if _TABLE is None:
        tab = np.empty((24, 49, 2, 2), dtype=object)
        for l in range(24):
            for d in range(49):
                for c in range(2):
                    for f in range(2):
                        tab[l, d, c, f] = _cluster(l, d, c, bool(f))
        lens = np.vectorize(len)(tab).astype(np.int64)
        _TABLE = (tab.reshape(-1), lens.reshape(-1))
    return _TABLE


_SEP = np.array(["", ",", "·", ":", ".", ";", ""], dtype=object)


def render(chars, boundary, dia, cap=None, punct=None):
    """-> (text, letter_offsets np.int64[n]). Letter ids >= 24 (mask/blank/pad) are not
    renderable and must not be passed. dia==48 / punct==6 (unknown) render as absent."""
    chars = np.asarray(chars, np.int64); n = len(chars)
    if n == 0:
        return "", np.zeros(0, np.int64)
    boundary = np.asarray(boundary, np.int64)
    dia = np.minimum(np.asarray(dia, np.int64), 48)
    cap = np.zeros(n, np.int64) if cap is None else np.asarray(cap, np.int64)
    punct = np.zeros(n, np.int64) if punct is None else np.minimum(np.asarray(punct, np.int64), 6)
    tab, lens = _table()
    word_end = (boundary == 1) | (boundary == 2)
    final = word_end.copy(); final[-1] = True
    key = ((chars * 49 + dia) * 2 + cap) * 2 + final
    space = word_end.copy(); space[-1] = False
    clus = tab[key]
    sep = _SEP[punct] + np.where(space, " ", "").astype(object)
    sep_len = (punct > 0) & (punct < 6)
    step = lens[key] + sep_len.astype(np.int64) + space.astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(step)[:-1]])
    text = "".join((clus + sep).tolist())
    return text, offs


def strip_diacritics(s: str) -> str:
    """Accent/breathing/iota-subscript/diaeresis stripped, case kept (used by the flat arm's
    diacritic-channel dropout)."""
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if not unicodedata.combining(c)))
