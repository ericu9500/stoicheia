"""HF-Hub-compatible processor for Stoicheia-meter: text <-> the model's five
input planes (chars/boundary/dia/punct/cap -- capitalization is a real input here,
unlike the base pretrained model, where it is output-only), plus decode helpers for
the two production tasks: macronization (vowel-length marks on ambiguous dichrona)
and metrical scansion (heavy/light/verse-final syllable bracketing).

Mark conventions and the ambiguous-dichrona rule are ported verbatim from
meter/marks.py (the training-side projection code) so a converted checkpoint's
predictions decode identically to the original project's `meter.predict` CLI.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MASK, BLANK, PAD = 24, 25, 26
UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6

ALPHABET = "αβγδεζηθικλμνξοπρστυφχψω"
LETTER_IDS = {c: i for i, c in enumerate(ALPHABET)}
ID2LETTER = np.array(list(ALPHABET))

_EXTRA_BASE = {
    "ς": "σ", "ϲ": "σ", "Ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϕ": "φ", "ϰ": "κ", "ϱ": "ρ", "ϖ": "π",
}
_MARK_MAP = {
    0x0301: "acute", 0x0341: "acute", 0x0300: "grave", 0x0340: "grave",
    0x0342: "circ", 0x0302: "circ", 0x0313: "smooth", 0x0343: "smooth",
    0x0314: "rough", 0x0345: "iota", 0x0308: "diaer",
}
_ACC = {"acute": 1, "grave": 2, "circ": 3}
_BR = {"smooth": 1, "rough": 2}


def _pack_dia(acc, br, iota, diaer):
    return ((acc * 3 + br) * 2 + iota) * 2 + diaer


def _unpack_dia(d):
    diaer = d % 2; d //= 2
    iota = d % 2; d //= 2
    br = d % 3; acc = d // 3
    return acc, br, iota, diaer


# ---- macron/scansion label conventions (meter/marks.py, reproduced verbatim) ----
MAC_LONG, MAC_SHORT = 0, 1
SCAN_O, SCAN_HEAVY, SCAN_LIGHT, SCAN_VERSE = 0, 1, 2, 3

_A, _E, _H, _I, _O, _Y, _W = (LETTER_IDS[c] for c in "αεηιουω")
DICHRONA_IDS = np.array([_A, _I, _Y])
VOWEL_IDS = np.array([_A, _E, _H, _I, _O, _Y, _W])
DIPHTHONGS = {(_A, _I), (_A, _Y), (_E, _I), (_E, _Y), (_H, _Y),
              (_O, _I), (_O, _Y), (_Y, _I), (_W, _Y)}


def ambiguous_mask(chars: np.ndarray, boundary: np.ndarray, dia: np.ndarray) -> np.ndarray:
    """Which plane positions are ambiguous dichrona (the macronizer's domain)? A
    position is ambiguous iff it's a base alpha/iota/upsilon without circumflex or
    iota subscript, and not part of a diphthong (diaeresis on the second vowel
    breaks the diphthong; pairs never span a word boundary)."""
    n = len(chars)
    d = np.asarray(dia, dtype=np.int64)
    acc, _br, iota, diaer = _unpack_dia(d.copy())
    is_dich = np.isin(chars, DICHRONA_IDS)
    out = is_dich & (acc != 3) & (iota == 0)
    if n > 1:
        chars = np.asarray(chars)
        pair = np.zeros(n - 1, dtype=bool)
        for f, s in DIPHTHONGS:
            pair |= (chars[:-1] == f) & (chars[1:] == s)
        pair &= np.asarray(boundary[:-1]) == 0
        out[1:] &= ~(pair & (diaer[1:] == 0))
        out[:-1] &= ~(pair & (diaer[1:] == 0))
    return out


def merge_vowelless_syllables(chars: np.ndarray, scan_labels: np.ndarray) -> np.ndarray:
    """A predicted syllable span with no vowel isn't a real syllable -- it's a
    boundary placed one letter early, typically at the first letter of a geminate
    consonant pair (e.g. "{λε}[ν]" for what should be one closed syllable
    "[λεν]"). Merge any such span into the preceding one, keeping its own weight
    label: that label (usually already correct, since it's typically a closing
    consonant) is normally right for the merged syllable -- only the boundary was
    misplaced. A vowel-less span at the very start of the line is left as-is."""
    out = np.asarray(scan_labels).copy()
    is_vowel = np.isin(np.asarray(chars), VOWEL_IDS)
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
    """A syllable containing a circumflexed vowel is always heavy -- a fixed rule
    of Greek prosody, not something the per-letter classifier can get wrong in
    principle, only in practice. Flip any SCAN_LIGHT span containing a circumflex
    to SCAN_HEAVY; SCAN_VERSE is left alone (it already renders as a heavy-looking
    bracket). Boundary placement itself is untouched -- this only corrects weight."""
    d = np.asarray(dia, dtype=np.int64)
    acc, _br, _iota, _diaer = _unpack_dia(d.copy())
    has_circ = acc == 3
    out = np.asarray(scan_labels).copy()
    start = 0
    for i in range(len(out)):
        if out[i] != SCAN_O:
            if out[i] == SCAN_LIGHT and has_circ[start:i + 1].any():
                out[i] = SCAN_HEAVY
            start = i + 1
    return out


def _is_letter(ch: str) -> bool:
    low = ch.lower()
    if low in LETTER_IDS or low in _EXTRA_BASE:
        return True
    dec = unicodedata.normalize("NFD", low)
    return bool(dec) and (dec[0] in LETTER_IDS or dec[0] in _EXTRA_BASE)


def insert_marks(plain: str, labels: dict) -> str:
    """Write `_` (long) / `^` (short) after the letters given by
    {letter_ordinal: MAC_LONG|MAC_SHORT} -- production macron output format."""
    nfc = unicodedata.normalize("NFC", plain)
    out, ordinal, pending = [], -1, None
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


def bracketize(plain: str, labels: dict) -> str:
    """Render per-letter scan labels back into [heavy]{light} syllable spans
    (verse-final span rendered as heavy, matching brevis-in-longo display)."""
    nfc = unicodedata.normalize("NFC", plain)
    out, cur, ordinal = [], [], -1
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


@dataclass
class _Encoded:
    chars: list
    boundary: list
    dia: list
    punct: list
    cap: list


class CharBertMeterProcessor:
    """`processor(text)` -> dict of batched tensors ready for `CharBertMeterModel(**batch)`."""

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def save_pretrained(self, save_directory, **_kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "processor_config.json").write_text(
            json.dumps({"processor_class": "CharBertMeterProcessor"}))

    def _classify(self, text: str) -> _Encoded:
        nfd = unicodedata.normalize("NFD", text)
        chars, boundary, dia, punct, cap = [], [], [], [], []
        acc = br = iota = diaer = 0
        pending_bnd = 0
        i = 0
        while i < len(nfd):
            ch = nfd[i]
            if ch == "-":
                run = 0
                while i < len(nfd) and nfd[i] == "-":
                    run += 1
                    i += 1
                for _ in range(run):
                    chars.append(MASK); boundary.append(UNK_BND)
                    dia.append(UNK_DIA); punct.append(UNK_PUNCT); cap.append(0)
                continue
            low = ch.lower()
            base = low if low in LETTER_IDS else _EXTRA_BASE.get(low)
            if base is not None:
                if chars and pending_bnd:
                    boundary[-1] = pending_bnd
                    pending_bnd = 0
                chars.append(LETTER_IDS[base])
                cap.append(1 if ch != low else 0)
                boundary.append(0)
                dia.append(0)
                punct.append(0)
                acc = br = iota = diaer = 0
            elif unicodedata.combining(ch) or ord(ch) in _MARK_MAP:
                kind = _MARK_MAP.get(ord(ch))
                if kind in _ACC:
                    acc = _ACC[kind]
                elif kind in _BR:
                    br = _BR[kind]
                elif kind == "iota":
                    iota = 1
                elif kind == "diaer":
                    diaer = 1
                if dia:
                    dia[-1] = _pack_dia(acc, br, iota, diaer)
            elif ch.isspace():
                pending_bnd = max(pending_bnd, 1)
            elif ch in ".;!?":
                pending_bnd = max(pending_bnd, 2)
                if punct:
                    punct[-1] = 4 if ch == "." else 5
            elif ch in ",:··":
                if punct:
                    punct[-1] = {",": 1, "·": 2, "·": 2, ":": 3}.get(ch, 0)
            i += 1
        if boundary:
            boundary[-1] = max(boundary[-1], 2)
        return _Encoded(chars, boundary, dia, punct, cap)

    def __call__(self, text: str, has_boundaries: bool = True):
        """Encode `text` into model-ready tensors. Unlike the base pretraining
        processor, `cap` is a real model input here (fine-tune-only channel)."""
        enc = self._classify(text)
        n = len(enc.chars)
        chars = np.array(enc.chars, dtype=np.int64)
        boundary = np.array(enc.boundary, dtype=np.int64)
        dia = np.array(enc.dia, dtype=np.int64)
        punct = np.array(enc.punct, dtype=np.int64)
        cap = np.array(enc.cap, dtype=np.int64)
        if not has_boundaries:
            boundary[:] = UNK_BND

        batch = dict(
            input_ids=torch.from_numpy(chars)[None],
            boundary=torch.from_numpy(boundary)[None],
            dia=torch.from_numpy(dia)[None],
            punct=torch.from_numpy(punct)[None],
            cap=torch.from_numpy(cap)[None],
            seg_id=torch.zeros(1, n, dtype=torch.long),
        )
        batch["_text"] = text  # kept out-of-band for decode (marks splice into the original string)
        batch["_chars"] = chars
        batch["_boundary"] = boundary
        batch["_dia"] = dia
        return batch

    # ---------------------------------------------------------------- decode

    def decode_macronization(self, model_out, batch) -> str:
        """Insert `_`/`^` (long/short) after every ambiguous alpha/iota/upsilon --
        matches `meter.predict --macronize` exactly (only ambiguous dichrona get a
        mark; unambiguous positions -- eta, omega, diphthongs, iota subscript,
        circumflexed vowels -- are left bare, since their length isn't in doubt)."""
        pred_mac = model_out.mac.argmax(-1)[0].numpy()
        amb = ambiguous_mask(batch["_chars"], batch["_boundary"], batch["_dia"])
        labels = {int(i): int(pred_mac[i]) for i in np.flatnonzero(amb)}
        return insert_marks(batch["_text"], labels)

    def decode_scansion(self, model_out, batch) -> str:
        """Bracket every syllable the model assigns a non-trivial weight to:
        [heavy], {light}, with the line-final syllable (brevis in longo) shown as
        heavy -- matches `meter.predict --scan` exactly, including its two
        deterministic corrections: merge_vowelless_syllables (a vowel-less
        predicted span gets folded into the preceding syllable) and
        enforce_circumflex_heavy (a circumflexed syllable is always heavy)."""
        pred_scan = model_out.scan.argmax(-1)[0].numpy()
        pred_scan = merge_vowelless_syllables(batch["_chars"], pred_scan)
        pred_scan = enforce_circumflex_heavy(batch["_dia"], pred_scan)
        labels = {i: int(c) for i, c in enumerate(pred_scan) if c > 0}
        return bracketize(batch["_text"], labels)
