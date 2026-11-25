"""Project macron / scansion annotations onto Stoicheia's letter planes.

The backbone codec (data/normalize.py) turns text into per-LETTER planes: every Greek
letter is one position; spaces, punctuation and editorial marks are folded into the
boundary/punct channels or stripped. Both annotation formats mark *letters*:

  macronized text   `_` (long) / `^` (short) written after an ambiguous dichronon,
                    e.g. "βα^ρύκτυ^πος"; combining macron/breve are accepted too
  bracketed verse   [heavy] {light} syllable spans, weight belonging to the last
                    letter of the span, the line-final syllable being verse-end
                    (brevis in longo), e.g. "[ὦ] [παῖ] {τέ}[λος] ..."

So a label is "letter ordinal -> class". We walk the annotated text with the SAME
character-kind LUT the codec uses, counting letters exactly as normalize_record will
count them on the stripped text — projection is alignment-exact by construction and
asserted at encode time.

Label conventions follow an existing macronization project (its Norma scorer and
scanner corpus are reused verbatim):
  macron: 0 = long, 1 = short
  scan:   0 = none, 1 = heavy syllable ends here, 2 = light ends, 3 = verse ends
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from meter.backbone import ALPHABET  # noqa: F401  (ensures STOICHEIA_ROOT is on sys.path)
from data.normalize import _KIND, K_LETTER, LETTER_IDS, unpack_dia

MAC_LONG, MAC_SHORT = 0, 1
SCAN_O, SCAN_HEAVY, SCAN_LIGHT, SCAN_VERSE = 0, 1, 2, 3
IGNORE = -100

# annotation characters (never part of the codec's letter set)
_LONG_MARKS = {"_", "̄"}   # ASCII underscore, combining macron
_SHORT_MARKS = {"^", "̆"}  # ASCII caret, combining breve
_ALL_MARKS = _LONG_MARKS | _SHORT_MARKS

_A, _E, _H, _I, _O, _Y, _W = (LETTER_IDS[c] for c in "αεηιουω")
DICHRONA_IDS = np.array([_A, _I, _Y])
VOWEL_IDS = np.array([_A, _E, _H, _I, _O, _Y, _W])
# (first, second) letter-id pairs that form a diphthong
DIPHTHONGS = {(_A, _I), (_A, _Y), (_E, _I), (_E, _Y), (_H, _Y),
              (_O, _I), (_O, _Y), (_Y, _I), (_W, _Y)}


def _is_letter(ch: str) -> bool:
    """Does this (possibly precomposed) character contribute one codec letter?"""
    cp = ord(unicodedata.normalize("NFD", ch)[0])
    return cp < len(_KIND) and _KIND[cp] == K_LETTER


def parse_macron_line(marked: str):
    """Annotated line -> (plain_text, {letter_ordinal: MAC_LONG|MAC_SHORT}).

    plain_text is the line with all length marks removed (NFC); letter ordinals
    count codec letters and therefore index normalize_record's planes directly.
    """
    nfd = unicodedata.normalize("NFD", marked)
    kept, labels = [], {}
    ordinal = -1
    for ch in nfd:
        if ch in _ALL_MARKS:
            if ordinal >= 0:
                labels[ordinal] = MAC_LONG if ch in _LONG_MARKS else MAC_SHORT
            continue
        cp = ord(ch)
        if cp < len(_KIND) and _KIND[cp] == K_LETTER:
            ordinal += 1
        kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), labels


_SYL = re.compile(r"\[([^\]]*)\]|\{([^}]*)\}")


def parse_scan_line(bracketed: str):
    """Bracketed verse -> (plain_text, {letter_ordinal: scan class}).

    The syllable weight sits on the LAST letter of the span (the old corpus puts it
    on the last non-space character, which can be an apostrophe — we take the last
    codec letter instead, which is what the planes can address). The final labeled
    letter of the line becomes SCAN_VERSE. Returns None for lines with no syllables.
    """
    plain_parts, labels = [], {}
    ordinal = -1
    pos = 0
    last_labeled = None

    def advance(text):
        nonlocal ordinal
        last = None
        for ch in text:
            if _is_letter(ch):
                ordinal += 1
                last = ordinal
        plain_parts.append(text)
        return last

    for m in _SYL.finditer(bracketed):
        advance(bracketed[pos:m.start()])
        text, weight = ((m.group(1), SCAN_HEAVY) if m.group(1) is not None
                        else (m.group(2), SCAN_LIGHT))
        last = advance(text)
        if last is not None:
            labels[last] = weight
            last_labeled = last
        pos = m.end()
    advance(bracketed[pos:])
    if last_labeled is None:
        return None
    labels[last_labeled] = SCAN_VERSE
    return unicodedata.normalize("NFC", "".join(plain_parts)), labels


def ambiguous_mask(chars: np.ndarray, boundary: np.ndarray, dia: np.ndarray):
    """Which plane positions are ambiguous dichrona (the macronizer's domain)?

    A position is ambiguous iff it is a base α/ι/υ that does not carry circumflex or
    iota subscript and is not part of a diphthong; diaeresis on the second vowel
    breaks the diphthong, and pairs never span a word boundary. Same rule as the old
    project's `markable()` (macronize_corpus.py), computed on the planes.
    """
    n = len(chars)
    d = np.asarray(dia, dtype=np.int64)
    acc, _br, iota, diaer = unpack_dia(d.copy())
    is_dich = np.isin(chars, DICHRONA_IDS)
    out = is_dich & (acc != 3) & (iota == 0)
    if n > 1:
        pair = np.zeros(n - 1, dtype=bool)
        for f, s in DIPHTHONGS:
            pair |= (chars[:-1] == f) & (chars[1:] == s)
        pair &= boundary[:-1] == 0          # no word boundary inside a diphthong
        # second element of a diphthong (unless it carries diaeresis)
        out[1:] &= ~(pair & (diaer[1:] == 0))
        # first element of a diphthong (unless the second carries diaeresis)
        out[:-1] &= ~(pair & (diaer[1:] == 0))
    return out


def merge_vowelless_syllables(chars: np.ndarray, scan_labels: np.ndarray) -> np.ndarray:
    """A predicted "syllable" span with no vowel isn't a syllable -- it's a boundary
    placed one letter early, typically at the first of a geminate consonant pair
    (e.g. predicted "{λε}[ν]" for what should be one closed syllable "[λεν]").
    Merge any such span into the PRECEDING one by dropping the earlier boundary,
    keeping the vowel-less span's OWN weight label: that label (usually SCAN_HEAVY,
    since it's a closing consonant) is normally already correct for the merged
    syllable -- only the boundary was misplaced. A vowel-less span at the very
    start of the line (no preceding syllable to merge into) is left as-is."""
    out = np.asarray(scan_labels).copy()
    is_vowel = np.isin(chars, VOWEL_IDS)
    kept = []
    start = 0
    for i in range(len(out)):
        if out[i] == SCAN_O:
            continue
        if not is_vowel[start:i + 1].any() and kept:
            out[kept.pop()] = SCAN_O
        kept.append(i)
        start = i + 1
    return out


def enforce_circumflex_heavy(dia: np.ndarray, scan_labels: np.ndarray) -> np.ndarray:
    """Circumflex marks a categorically long vowel, and a syllable containing one is
    always heavy -- a fixed rule of Greek prosody, not something the per-letter scan
    classifier can get wrong in principle, only in practice. Walk each predicted
    syllable span (consecutive letters up to and including the next non-SCAN_O
    label); if it contains a circumflexed letter and the model called it
    SCAN_LIGHT, flip that span's label to SCAN_HEAVY. SCAN_VERSE is left alone (it
    already renders as a heavy-looking bracket); boundary PLACEMENT is untouched --
    this only corrects a syllable's weight, never whether one was predicted there."""
    d = np.asarray(dia, dtype=np.int64)
    acc, _br, _iota, _diaer = unpack_dia(d.copy())
    has_circ = acc == 3
    out = np.asarray(scan_labels).copy()
    start = 0
    for i in range(len(out)):
        if out[i] != SCAN_O:
            if out[i] == SCAN_LIGHT and has_circ[start:i + 1].any():
                out[i] = SCAN_HEAVY
            start = i + 1
    return out


def insert_marks(plain: str, labels: dict[int, int]) -> str:
    """Write `_`/`^` after the letters given by {letter_ordinal: MAC_*} (production
    output format, identical to the old project's)."""
    nfc = unicodedata.normalize("NFC", plain)
    out = []
    ordinal = -1
    pending = None
    for ch in nfc:
        if pending is not None and not unicodedata.category(ch).startswith("M"):
            out.append(pending)
            pending = None
        out.append(ch)
        if _is_letter(ch):
            ordinal += 1
            if ordinal in labels:
                pending = "_" if labels[ordinal] == MAC_LONG else "^"
    if pending is not None:
        out.append(pending)
    return "".join(out)


def bracketize(plain: str, labels: dict[int, int]) -> str:
    """Render per-letter scan labels back into [heavy]{light} spans (verse-end span
    is emitted as heavy, matching brevis in longo display in the old corpus)."""
    nfc = unicodedata.normalize("NFC", plain)
    out, cur = [], []
    ordinal = -1
    for ch in nfc:
        cur.append(ch)
        if _is_letter(ch):
            ordinal += 1
            lab = labels.get(ordinal, SCAN_O)
            if lab != SCAN_O:
                o, c = ("{", "}") if lab == SCAN_LIGHT else ("[", "]")
                out.append(o + "".join(cur) + c)
                cur = []
    if cur:
        out.append("".join(cur))
    return "".join(out)
