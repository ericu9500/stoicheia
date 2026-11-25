"""Normalize polytonic Ancient Greek into the model's char stream + label planes.

Input:  NFC polytonic text (spaces, punctuation, capitals, diacritics).
Output: per record, four aligned uint8 arrays of equal length N (one entry per letter):
  chars     letter id in the minimal alphabet (alpha..omega, final/lunate sigma merged)
  boundary  0=word-internal  1=word-final  2=sentence-final
  dia       ((accent*3 + breathing)*2 + iota_sub)*2 + diaeresis   in [0, 48)
            accent: 0=none 1=acute 2=grave 3=circumflex; breathing: 0=none 1=smooth 2=rough
  cap       0/1 capitalized letter

Everything stripped (punctuation, spaces, Latin, digits, sigla, exotic marks) is counted in
`Stats` so nothing disappears silently. The transform is exactly invertible to the spaced,
punctuated-by-category, accentless source (see tests/test_normalize.py round-trip property).
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- alphabet

ALPHABET = "αβγδεζηθικλμνξοπρστυφχψω"  # 24 letters; sigma merged
LETTER_IDS = {c: i for i, c in enumerate(ALPHABET)}
ID2LETTER = np.array(list(ALPHABET))

# Codepoints that map onto the 24 letters after lowercasing (NFD base letters only).
_EXTRA_BASE = {
    "ς": "σ",       # final sigma
    "ϲ": "σ",       # lunate sigma
    "Ϲ": "σ",       # capital lunate sigma
    "ϐ": "β",       # curled beta
    "ϑ": "θ",       # theta symbol
    "ϕ": "φ",       # phi symbol
    "ϰ": "κ",       # kappa symbol
    "ϱ": "ρ",       # rho symbol
    "ϖ": "π",       # pi symbol
}
# Archaic/numeral letters: counted, then stripped (frequency report decides their fate).
ARCHAIC = set("ϝϜϙϘϟϞϡϠϛϚͱͰͳͲϻϺϸϷ")

# ---------------------------------------------------------------- kind LUTs

K_STRIP, K_LETTER, K_MARK, K_SPACE, K_WPUNCT, K_SPUNCT, K_LATIN, K_DIGIT, K_ARCHAIC = range(9)

# combining marks we model
M_ACUTE, M_GRAVE, M_CIRC, M_SMOOTH, M_ROUGH, M_IOTA, M_DIAER, M_OTHER = range(8)

_MARK_MAP = {
    0x0301: M_ACUTE, 0x0341: M_ACUTE,          # oxia folds to acute
    0x0300: M_GRAVE, 0x0340: M_GRAVE,          # varia folds to grave
    0x0342: M_CIRC,                            # perispomeni
    0x0302: M_CIRC,                            # circumflex accent (rare in grc)
    0x0313: M_SMOOTH, 0x0343: M_SMOOTH,        # psili / koronis
    0x0314: M_ROUGH,                           # dasia
    0x0345: M_IOTA,                            # ypogegrammeni
    0x0308: M_DIAER,                           # dialytika
    # stripped-but-counted marks
    0x0304: M_OTHER, 0x0306: M_OTHER,          # macron, breve
    0x0323: M_OTHER,                           # dot below (editorial uncertainty)
    0x0331: M_OTHER, 0x0345 + 0x10000: M_OTHER,  # placeholder, unreachable
}

_SPUNCT = set(".;!?") | {";", "؟"}          # ; and Greek question mark = sentence end
_WPUNCT = (
    set(",:·«»\"'()[]{}<>—–-‐‒†‡*⟨⟩⌈⌋⌊⌉|/\\_=+~^%$#@&")
    | {"’", "‘", "᾽", "ʼ", "῾", "᾿",  # apostrophes/koronis forms
       "«", "»", "“", "”", "…", "·", "·",
       "†", "‡", "⸎"}
)

_LUT_MAX = 0x20000


def _build_luts():
    kind = np.zeros(_LUT_MAX, dtype=np.uint8)  # K_STRIP default
    base = np.zeros(_LUT_MAX, dtype=np.uint8)
    capf = np.zeros(_LUT_MAX, dtype=np.uint8)
    mark = np.full(_LUT_MAX, 255, dtype=np.uint8)

    for cp in range(_LUT_MAX):
        ch = chr(cp)
        # decomposed base letters live in 0x0370-0x03FF after NFD
        low = ch.lower()
        if low in LETTER_IDS:
            kind[cp] = K_LETTER
            base[cp] = LETTER_IDS[low]
            capf[cp] = 1 if ch != low else 0
        elif low in _EXTRA_BASE:
            kind[cp] = K_LETTER
            base[cp] = LETTER_IDS[_EXTRA_BASE[low]]
            capf[cp] = 1 if (ch != low or ch == "Ϲ") else 0
        elif ch in ARCHAIC:
            kind[cp] = K_ARCHAIC
        elif cp in _MARK_MAP:
            kind[cp] = K_MARK
            mark[cp] = _MARK_MAP[cp]
        elif unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            kind[cp] = K_MARK
            mark[cp] = M_OTHER
        elif ch.isspace():
            kind[cp] = K_SPACE
        elif ch in _SPUNCT:
            kind[cp] = K_SPUNCT
        elif ch in _WPUNCT:
            kind[cp] = K_WPUNCT
        elif "a" <= low <= "z":
            kind[cp] = K_LATIN
        elif ch.isdigit():
            kind[cp] = K_DIGIT
        elif unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S"):
            kind[cp] = K_WPUNCT
    # punctuation class per codepoint: 0 none, 1 comma, 2 high-dot(·), 3 colon, 4 period, 5 question
    punct = np.zeros(_LUT_MAX, dtype=np.uint8)
    for c, cls in ((",", 1), ("·", 2), ("·", 2), ("·", 2), (":", 3),
                   (".", 4), (";", 5), (";", 5), ("?", 5), ("!", 5)):
        punct[ord(c)] = cls
    return kind, base, capf, mark, punct


_KIND, _BASE, _CAPF, _MARK, _PUNCT = _build_luts()
N_PUNCT = 6

# separator "boundary weight": what a between-letters char implies for the previous letter
#           STRIP LETTER MARK SPACE WPUNCT SPUNCT LATIN DIGIT ARCHAIC
_BWEIGHT = np.array([0, 0, 0, 1, 1, 2, 1, 1, 1], dtype=np.uint8)


@dataclass
class Stats:
    records_in: int = 0
    records_kept: int = 0
    records_dropped_nongreek: int = 0
    records_dropped_empty: int = 0
    letters: int = 0
    words: int = 0
    sentences: int = 0
    stripped: Counter = field(default_factory=Counter)     # kind -> count
    archaic: Counter = field(default_factory=Counter)      # char -> count
    other_marks: Counter = field(default_factory=Counter)  # codepoint hex -> count
    mark_conflicts: int = 0
    orphan_marks: int = 0

    def merge(self, o: "Stats"):
        for k in ("records_in", "records_kept", "records_dropped_nongreek",
                  "records_dropped_empty", "letters", "words", "sentences",
                  "mark_conflicts", "orphan_marks"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.stripped.update(o.stripped)
        self.archaic.update(o.archaic)
        self.other_marks.update(o.other_marks)


DIA_STATES = 48


def _pack_dia(acc, br, iota, diaer):
    return ((acc * 3 + br) * 2 + iota) * 2 + diaer


def unpack_dia(d):
    diaer = d % 2; d //= 2
    iota = d % 2; d //= 2
    br = d % 3; acc = d // 3
    return acc, br, iota, diaer


MIN_GREEK_RATIO = 0.95


def normalize_record(text: str, stats: Stats, with_punct=False):
    """Return (chars, boundary, dia, cap) uint8 arrays, or None if the record is dropped."""
    stats.records_in += 1
    nfd = unicodedata.normalize("NFD", text)
    cp = np.frombuffer(nfd.encode("utf-32-le"), dtype=np.uint32)
    cp = np.where(cp < _LUT_MAX, cp, 0)

    kind = _KIND[cp]
    letters = kind == K_LETTER
    n = int(letters.sum())

    n_latin = int((kind == K_LATIN).sum())
    if n == 0 or (n / max(n + n_latin, 1)) < MIN_GREEK_RATIO:
        if n == 0:
            stats.records_dropped_empty += 1
        else:
            stats.records_dropped_nongreek += 1
        return None

    lpos = np.flatnonzero(letters)                       # positions of letters in cp
    chars = _BASE[cp[lpos]]
    cap = _CAPF[cp[lpos]]

    # letter ordinal at every position (= index of previous-or-current letter)
    lord = np.cumsum(letters) - 1                        # -1 before first letter

    # ---- diacritics: marks attach to the preceding letter
    marks = np.flatnonzero(kind == K_MARK)
    acc = np.zeros(n, dtype=np.uint8)
    br = np.zeros(n, dtype=np.uint8)
    iota = np.zeros(n, dtype=np.uint8)
    diaer = np.zeros(n, dtype=np.uint8)
    if marks.size:
        tgt = lord[marks]
        ok = tgt >= 0
        stats.orphan_marks += int((~ok).sum())
        marks, tgt = marks[ok], tgt[ok]
        mk = _MARK[cp[marks]]
        for arr, kinds, vals in (
            (acc, (M_ACUTE, M_GRAVE, M_CIRC), (1, 2, 3)),
            (br, (M_SMOOTH, M_ROUGH), (1, 2)),
        ):
            for mkind, val in zip(kinds, vals):
                sel = mk == mkind
                if sel.any():
                    prev = arr[tgt[sel]]
                    stats.mark_conflicts += int(((prev != 0) & (prev != val)).sum())
                    arr[tgt[sel]] = val
        iota[tgt[mk == M_IOTA]] = 1
        diaer[tgt[mk == M_DIAER]] = 1
        other = mk == M_OTHER
        if other.any():
            for c in np.unique(cp[marks[other]]):
                stats.other_marks[f"U+{c:04X}"] += int((cp[marks[other]] == c).sum())
    dia = _pack_dia(acc.astype(np.int16), br, iota, diaer).astype(np.uint8)

    # ---- boundaries: max separator weight between letter i and letter i+1
    w = _BWEIGHT[kind]
    boundary = np.zeros(n, dtype=np.uint8)
    if n > 1:
        # cumulative max trick: segment-max of w over (lpos[i], lpos[i+1]) for each gap
        cw = np.maximum.reduceat(np.concatenate([w, [0]]), lpos)  # max over [lpos[i], lpos[i+1])
        # reduceat includes the letter itself at lpos[i] (weight 0) — harmless
        boundary[:-1] = cw[:-1]
    boundary[-1] = 2                                     # record end = sentence end
    # trailing separators after last letter may still say "sentence"
    if lpos[-1] + 1 < len(w) and w[lpos[-1] + 1:].size and w[lpos[-1] + 1:].max() >= 2:
        boundary[-1] = 2

    # ---- punctuation class per letter: which mark (if any) follows this letter in the gap
    #      (comma/high-dot/colon/period/question). Same segment-max trick as boundary.
    punct = np.zeros(n, dtype=np.uint8)
    if n > 1:
        pw = _PUNCT[cp]
        cpu = np.maximum.reduceat(np.concatenate([pw, [0]]), lpos)
        punct[:-1] = cpu[:-1]
    if lpos[-1] + 1 < len(cp):
        tail = _PUNCT[cp[lpos[-1] + 1:]]
        if tail.size:
            punct[-1] = tail.max()

    # ---- stats
    stats.records_kept += 1
    stats.letters += n
    stats.words += int((boundary >= 1).sum())
    stats.sentences += int((boundary == 2).sum())
    for k in (K_SPACE, K_WPUNCT, K_SPUNCT, K_LATIN, K_DIGIT, K_STRIP):
        c = int((kind == k).sum())
        if c:
            stats.stripped[k] += c
    if (kind == K_ARCHAIC).any():
        for c in np.unique(cp[kind == K_ARCHAIC]):
            stats.archaic[chr(c)] += int((cp[kind == K_ARCHAIC] == c).sum())

    if with_punct:
        return chars, boundary, dia, cap, punct
    return chars, boundary, dia, cap


# ---------------------------------------------------------------- inverses

def denormalize(chars: np.ndarray, boundary: np.ndarray) -> str:
    """Accentless spaced text; '. ' marks sentence-final words."""
    out = []
    for c, b in zip(chars, boundary):
        out.append(ID2LETTER[c])
        if b == 1:
            out.append(" ")
        elif b == 2:
            out.append(". ")
    return "".join(out).rstrip()


_MARK_CHARS = {M_ACUTE: "́", M_GRAVE: "̀", M_CIRC: "͂",
               M_SMOOTH: "̓", M_ROUGH: "̔", M_IOTA: "ͅ", M_DIAER: "̈"}
_ACC_M = {1: M_ACUTE, 2: M_GRAVE, 3: M_CIRC}
_BR_M = {1: M_SMOOTH, 2: M_ROUGH}


def restore_polytonic(chars, dia, cap, boundary) -> list[str]:
    """Reconstruct NFC polytonic words (final sigma reinstated) from the planes."""
    words, cur = [], []
    n = len(chars)
    for i in range(n):
        ch = ID2LETTER[chars[i]]
        acc, br, io, dd = unpack_dia(int(dia[i]))
        if cap[i]:
            ch = ch.upper()
        s = ch
        if br:
            s += _MARK_CHARS[_BR_M[br]]
        if dd:
            s += _MARK_CHARS[M_DIAER]
        if acc:
            s += _MARK_CHARS[_ACC_M[acc]]
        if io:
            s += _MARK_CHARS[M_IOTA]
        cur.append(s)
        if boundary[i] >= 1:
            w = "".join(cur)
            if w and w[-1] == "σ":
                w = w[:-1] + "ς"
            words.append(unicodedata.normalize("NFC", w))
            cur = []
    if cur:
        w = "".join(cur)
        if w and w[-1] == "σ":
            w = w[:-1] + "ς"
        words.append(unicodedata.normalize("NFC", w))
    return words
