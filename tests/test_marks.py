"""Label-projection correctness: alignment round-trips on real corpus lines, and the
ambiguity mask checked against the old project's `markable()` (ported verbatim)."""
import json
import os
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from meter.dataset import (concat_verses, encode_macron_line, encode_plain,
                           encode_scan_line)
from meter.marks import (MAC_LONG, MAC_SHORT, SCAN_HEAVY, SCAN_LIGHT, SCAN_VERSE,
                         ambiguous_mask, bracketize, enforce_circumflex_heavy,
                         insert_marks, merge_vowelless_syllables,
                         parse_macron_line, parse_scan_line)

SRC = Path(os.path.expandvars(os.environ.get("MACRONIZER_SRC",
                          "$MACRONIZER_SRC")))

# The four tests below read the released meter-silver tree (MACRONIZER_SRC). Skip them
# rather than fail when it is not checked out: the rest of this file is pure unit tests.
needs_src = pytest.mark.skipif(
    not (SRC / "data").is_dir(),
    reason="MACRONIZER_SRC not set to a checkout of the meter-silver dataset")

# ---------------------------------------------------------------- old-project reference
DICHRONA = set("αιυ")
DIPHTHONGS = {"αι", "αυ", "ει", "ευ", "ηυ", "οι", "ου", "υι", "ωυ"}
PERISPOMENI, YPOGEGRAMMENI, DIAERESIS = "͂", "ͅ", "̈"


def _base(ch):
    return unicodedata.normalize("NFD", ch)[0].lower()


def _has(ch, mark):
    return mark in unicodedata.normalize("NFD", ch)


def markable_ref(chars, i):
    """Verbatim port of the macron data tree/scripts/macronize_corpus.py::markable."""
    ch = chars[i]
    b = _base(ch)
    if b not in DICHRONA:
        return False
    if _has(ch, PERISPOMENI) or _has(ch, YPOGEGRAMMENI):
        return False
    if i > 0 and not _has(ch, DIAERESIS) and _base(chars[i - 1]) + b in DIPHTHONGS:
        return False
    if (i + 1 < len(chars) and not _has(chars[i + 1], DIAERESIS)
            and b + _base(chars[i + 1]) in DIPHTHONGS):
        return False
    return True


# ---------------------------------------------------------------- macron parsing

def test_parse_macron_simple():
    plain, labels = parse_macron_line("ὦ παῖ τέλος μὲν Ζεὺς ἔχει βα^ρύκτυ^πος")
    assert plain == "ὦ παῖ τέλος μὲν Ζεὺς ἔχει βαρύκτυπος"
    # letters: ω π α ι τ ε λ ο σ μ ε ν ζ ε υ σ ε χ ε ι β α(21) ρ υ κ τ υ(26) π ο σ
    assert labels == {21: MAC_SHORT, 26: MAC_SHORT}


def test_parse_macron_combining_marks():
    plain, labels = parse_macron_line("βᾱρῠ́ς")   # combining macron + breve-with-acute
    assert plain == "βαρύς"
    assert labels == {1: MAC_LONG, 3: MAC_SHORT}   # β0 α1 ρ2 υ3 ς4


@needs_src
def test_macron_roundtrip_against_plain_column():
    """TSV col1 (plain) and col2 (marked) must strip to identical letter streams."""
    checked = 0
    for name in ("hypotactic", "oga_0", "anthology", "theocritus_doric"):
        path = SRC / "data" / f"{name}.tsv"
        with open(path, encoding="utf-8") as f:
            for _ in range(300):
                line = f.readline()
                if not line:
                    break
                plain_col, marked = line.rstrip("\n").split("\t")[:2]
                plain, labels = parse_macron_line(marked)
                r1, r2 = encode_plain(plain), encode_plain(plain_col)
                if r1 is None or r2 is None:
                    continue
                assert np.array_equal(r1.chars, r2.chars), (name, marked)
                if labels:
                    assert max(labels) < len(r1.chars)
                checked += 1
    assert checked > 850   # theocritus_doric has only 18 lines


def test_insert_marks_roundtrip():
    for marked in ("ἥσθην δὲ βαιά^, πά^νυ^ δὲ βαιά^, τέττα^ρα^·",
                   "Δάφνι τά_λαν, τί_ τὺ_ τά_κεαι, ἁ_ δέ τε κώρα",
                   "χρὴ γι^νώσκειν ὅτι^ πά_σης τῆς γῆς ὁ περί^μετρος 0 ."):
        plain, labels = parse_macron_line(marked)
        again = insert_marks(plain, labels)
        assert parse_macron_line(again) == (plain, labels)
        assert unicodedata.normalize("NFC", again) == unicodedata.normalize("NFC", marked)


# ---------------------------------------------------------------- ambiguity mask

def _mask_via_planes(text):
    rec = encode_plain(text)
    return rec, ambiguous_mask(rec.chars, rec.boundary, rec.dia)


@needs_src
def test_ambiguous_mask_matches_reference():
    lines = []
    for name in ("hypotactic", "oga_1", "anthology", "drama_ia6"):
        with open(SRC / "data" / f"{name}.tsv", encoding="utf-8") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                lines.append(line.split("\t")[0])
    checked = 0
    for text in lines:
        text = unicodedata.normalize("NFC", text)
        rec = encode_plain(text)
        if rec is None:
            continue
        # reference mask over raw chars, projected to letter ordinals
        chars = list(text)
        ref = []
        for i, ch in enumerate(chars):
            if _base(ch).lower() in set("αβγδεζηθικλμνξοπρστυφχψω") | {"ς", "ϲ"}:
                ref.append(markable_ref(chars, i))
        if len(ref) != len(rec.chars):
            continue  # letters the raw walk counts differently (archaic etc.) — rare
        ours = ambiguous_mask(rec.chars, rec.boundary, rec.dia)
        assert ref == ours.tolist(), text
        checked += 1
    assert checked > 500


# ---------------------------------------------------------------- scanner parsing

def test_parse_scan_simple():
    line = "[ὦ] [παῖ] {τέ}[λος] [μὲν] [Ζεὺ]{ς ἔ}[χει] {βα}[ρύκ]{τυ}[πος]"
    plain, labels = parse_scan_line(line)
    assert plain == "ὦ παῖ τέλος μὲν Ζεὺς ἔχει βαρύκτυπος"
    # letter ordinals:  ὦ=0 π1 α2 ι3 τ4 έ5 λ6 ο7 ς8 μ9 ὲ10 ν11 Ζ12 ε13 ὺ14 ς15
    #                   ἔ16 χ17 ε18 ι19 β20 α21 ρ22 ύ23 κ24 τ25 υ26 π27 ο28 ς29
    assert labels[0] == SCAN_HEAVY and labels[3] == SCAN_HEAVY
    assert labels[5] == SCAN_LIGHT and labels[8] == SCAN_HEAVY
    assert labels[14] == SCAN_HEAVY and labels[16] == SCAN_LIGHT  # [Ζεὺ] ends at ὺ
    assert labels[29] == SCAN_VERSE
    assert max(labels) == 29


@needs_src
def test_scan_corpus_lines_encode():
    ok = 0
    with open(SRC / "data/scanner/corpus_v3.tsv", encoding="utf-8") as f:
        for _ in range(500):
            line = f.readline()
            if not line:
                break
            work, _meter, bracketed = line.rstrip("\n").split("\t")
            rec = encode_scan_line(bracketed)
            if rec is None:
                continue
            ends = (rec.y_scan > 0).sum()
            assert (rec.y_scan == SCAN_VERSE).sum() == 1
            assert ends >= 2, bracketed
            ok += 1
    assert ok > 450


def test_concat_verses_boundaries():
    r1 = encode_scan_line("[ὦ] [παῖ] {τέ}[λος]")
    r2 = encode_scan_line("{βα}[ρύκ]{τυ}[πος]")
    joined = concat_verses([r1, r2])
    n1 = len(r1)
    assert joined.boundary[n1 - 1] == 1        # seam demoted to word boundary
    assert joined.boundary[-1] == 2            # record end keeps sentence boundary
    assert (joined.y_scan == SCAN_VERSE).sum() == 2


def test_bracketize_roundtrip():
    line = "[ὦ] [παῖ] {τέ}[λος] [μὲν] [Ζεὺ]{ς ἔ}[χει] {βα}[ρύκ]{τυ}[πος]"
    plain, labels = parse_scan_line(line)
    out = bracketize(plain, {k: v for k, v in labels.items()})
    plain2, labels2 = parse_scan_line(out)
    assert plain2 == plain
    assert labels2 == labels


def test_enforce_circumflex_heavy_overrides_light():
    # "πᾶς" (circumflex on alpha, closed by sigma): a syllable containing a
    # circumflex is always heavy in Greek prosody, regardless of what the
    # per-letter classifier predicted.
    rec = encode_plain("πᾶς")
    labels = np.zeros(len(rec.chars), dtype=np.int64)
    labels[-1] = SCAN_LIGHT   # simulates the model's wrong prediction
    fixed = enforce_circumflex_heavy(rec.dia, labels)
    assert fixed[-1] == SCAN_HEAVY


def test_enforce_circumflex_heavy_leaves_non_circumflex_alone():
    rec = encode_plain("πολις")
    labels = np.zeros(len(rec.chars), dtype=np.int64)
    labels[-1] = SCAN_LIGHT
    fixed = enforce_circumflex_heavy(rec.dia, labels)
    assert fixed[-1] == SCAN_LIGHT


def test_merge_vowelless_syllables():
    # "{λε}[ν]" -> "[λεν]": a vowel-less span ("ν" alone) can't be a real
    # syllable -- fold it into the preceding one, keeping its own weight.
    plain, labels = parse_scan_line("{λε}[ν]")
    rec = encode_plain(plain)
    arr = np.zeros(len(rec.chars), dtype=np.int64)
    for k, v in labels.items():
        arr[k] = v
    fixed = merge_vowelless_syllables(rec.chars, arr)
    out = bracketize(plain, {i: int(l) for i, l in enumerate(fixed) if l})
    assert out == "[λεν]"


def test_merge_vowelless_syllables_leaves_real_syllables_alone():
    plain, labels = parse_scan_line("[ὦ] [παῖ]")
    rec = encode_plain(plain)
    arr = np.zeros(len(rec.chars), dtype=np.int64)
    for k, v in labels.items():
        arr[k] = v
    fixed = merge_vowelless_syllables(rec.chars, arr)
    assert fixed.tolist() == arr.tolist()


# ---------------------------------------------------------------- norma gold

@needs_src
def test_norma_lines_parse():
    n_mac = n_syl = 0
    with open(SRC / "data/norma/test.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["task"] == "macronize":
                plain, labels = parse_macron_line(d["text"])
                rec = encode_plain(plain)
                if labels:
                    assert rec is not None and max(labels) < len(rec.chars), d
                n_mac += 1
            else:
                parsed = parse_scan_line(d["text"])
                assert parsed is not None, d
                plain, labels = parsed
                rec = encode_plain(plain)
                assert rec is not None and max(labels) < len(rec.chars), d
                n_syl += 1
    assert n_mac == 932 and n_syl == 932
