"""I.PHI (Ithaca inscriptions) loader — segments, splits, planes.

Reads raw/iphi.jsonl (Ericu950/Inscriptions_2). Split rule (matches Ithaca):
PHI_ID last digit 3 -> test, 4 -> val, else train.

Improvement over the grc-encoder pilot loader: inscriptions contain runs of '-' marking
LOST characters. Normalizing straight through silently closes those gaps, gluing text
across real lacunae into false contexts. Here each inscription is SPLIT at '-' runs into
segments of continuous known text; training and eval only ever see genuine contexts, and
eval gaps always have known gold.
"""
from __future__ import annotations

import json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np

from data.normalize import ALPHABET, Stats, normalize_record
try:                    # imported as a package module (python -m insc.data.iphi)
    from insc.data.meta_vocab import region_to_id, record_century_id
except ImportError:     # imported with insc/data on sys.path (how the trainers load it)
    from meta_vocab import region_to_id, record_century_id

JSONL = Path(os.path.expandvars("$INS_DATA/raw/iphi.jsonl"))
GAP_RE = re.compile(r"-+")

ALIST = list(ALPHABET)
A_IDX = {c: i for i, c in enumerate(ALIST)}
MASK, UNK_BND, UNK_DIA, UNK_PUNCT = 24, 3, 48, 6

# Leiden-markup line-break extraction from the `edition` field(also used by the restoration evals
# too, duplicated here rather than imported to keep insc_data/ and insc_eval/ independent).
_TAG_RE = re.compile(r"<[^>]+>")
_ANGLE_RE = re.compile(r"&lt;|&gt;")
_CURLY_RE = re.compile(r"\{[^{}]*\}")
_BRACKET_STRIP_RE = re.compile(r"[\[\]]")


def line_break_ordinals(edition):
    """`edition` field (Leiden markup, '|' = line break) -> (line_ends, n_ordinal) where
    line_ends is a sorted list of LETTER-ordinal positions (0-indexed, counting each real
    letter AND each '-' as one lost-letter position -- matching how text_to_full_planes's
    chars array counts positions) immediately BEFORE which a '|' occurred, and n_ordinal is
    the total count. Bracket-restored letters count as normal letters (the brackets
    themselves are stripped, not their content); curly-brace deletions/footnotes and HTML
    tags are dropped entirely (never counted), matching phi_disagree.py's parse_record.
    Caller must cross-validate n_ordinal against the corresponding with_diacritics-based
    record's own length before trusting these positions -- edition and with_diacritics are
    independently-formatted views of the same edition and can disagree (OCR/encoding
    differences, a genuinely different field revision, etc.)."""
    s = _TAG_RE.sub(" ", edition)
    s = _ANGLE_RE.sub("", s)
    s = _CURLY_RE.sub("", s)
    s = _BRACKET_STRIP_RE.sub("", s)
    ordinal = 0
    line_ends = []
    for ch in s:
        if ch == "|":
            line_ends.append(ordinal)
        elif ch == "-":
            ordinal += 1
        elif ch.isspace():
            continue
        else:
            base = unicodedata.normalize("NFD", ch)[0]
            cp = ord(base)
            if (0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF) and \
                    unicodedata.category(base).startswith("L"):
                ordinal += 1
    return line_ends, ordinal


def text_to_planes(t):
    """ithaca_text (lowercase, accentless, spaces, '-' damage runs) -> (chars, boundary)
    arrays, WHOLE text, damage kept in place as MASK positions -- NOT split into
    segments. Mirrors insc_eval/restore_strict.py's text_to_planes() (duplicated here,
    not imported, to keep insc_data/ and insc_eval/ independent of each other)."""
    ids, bnd = [], []
    for ch in t:
        if ch == " ":
            if bnd:
                bnd[-1] = 1
        elif ch == "-":
            ids.append(MASK); bnd.append(UNK_BND)
        elif ch in A_IDX:
            ids.append(A_IDX[ch]); bnd.append(0)
    return np.array(ids, np.int64), np.array(bnd, np.int64)


def split_of(phi_id):
    s = str(phi_id).strip()
    if not s or not s[-1].isdigit():
        return "train"
    test_d = os.environ.get("INSC_TEST_DIGIT", "3")
    val_d = os.environ.get("INSC_VAL_DIGIT", "4")
    return {val_d: "val", test_d: "test"}.get(s[-1], "train")


def load(split=None, min_len=32, field="with_diacritics", max_records=None):
    """Yield dicts: chars/boundary/dia/cap/punct planes + phi_id/split/region/tpq/taq.
    One dict per continuous SEGMENT (inscriptions split at '-' lacuna runs)."""
    out = []
    st = Stats()
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        sp = split_of(r.get("PHI_ID"))
        if split and sp != split:
            continue
        # fallback to ithaca_text only for the primary field — synthetic columns must
        # never silently substitute the real text
        text = (r.get(field) or (r.get("ithaca_text") if field == "with_diacritics" else "")) or ""
        for seg_i, seg in enumerate(GAP_RE.split(text)):
            if len(seg.strip()) < min_len:
                continue
            nr = normalize_record(seg, st, with_punct=True)
            if nr is None or len(nr[0]) < min_len:
                continue
            chars, boundary, dia, cap, punct = nr
            region = r.get("main_region"); tpq = r.get("tpq"); taq = r.get("taq")
            out.append(dict(
                chars=chars, boundary=boundary, dia=dia, cap=cap, punct=punct,
                phi_id=r.get("PHI_ID"), seg=seg_i, split=sp,
                region=region, tpq=tpq, taq=taq,
                region_id=region_to_id(region), century_id=record_century_id(tpq, taq)))
            if max_records and len(out) >= max_records:
                return out
    return out


def load_whole(split=None, min_len=32, max_len=None, field="ithaca_text", max_records=None):
    """Yield dicts: chars/boundary planes + phi_id/split/region/tpq/taq -- ONE dict per
    WHOLE inscription, damage ('-' runs) kept in place as MASK positions rather than
    split away. Use this (not load()) for anything that needs to match how the
    inscription is actually evaluated end-to-end (e.g. attribution): a model trained
    only on load()'s damage-free segments never sees a mid-sequence gap during
    training, which is a real train/test distribution mismatch against real,
    frequently-damaged inscriptions (~52% of the I.PHI test population contains a gap).
    """
    out = []
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        sp = split_of(r.get("PHI_ID"))
        if split and sp != split:
            continue
        text = " ".join((r.get(field) or "").strip().lower().split())
        if len(text) < min_len or (max_len and len(text) > max_len):
            continue
        chars, boundary = text_to_planes(text)
        if len(chars) < min_len:
            continue
        out.append(dict(chars=chars, boundary=boundary,
                        phi_id=r.get("PHI_ID"), seg=0, split=sp,
                        region=r.get("main_region"), tpq=r.get("tpq"), taq=r.get("taq")))
        if max_records and len(out) >= max_records:
            return out
    return out


def text_to_full_planes(text, stats=None):
    """Raw accented/punctuated text (with '-' damage runs) -> (chars, boundary, dia, cap,
    punct, is_real_lacuna) arrays, WHOLE text, damage kept in place as MASK+unknown
    positions rather than split away or stripped. Reusable per-text encoder shared by
    load_whole_full() and any eval script that needs to feed our model its natural full
    representation (real accents/breathings/punctuation) for the SAME underlying text
    Ithaca sees in its own reduced deaccented-but-spaced format. Returns None if
    normalize_record() rejects any non-gap span.

    is_real_lacuna marks positions where the true content is GENUINELY unknown (a real
    '-' run from the edition itself, not a synthetic training mask) -- downstream noising
    must never select these for additional synthetic damage and must never supervise a
    label there (no ground truth exists, unlike a synthetically-masked span over known
    text). cap has no input channel at all (model/char_bert.py's forward() never reads a
    'cap' key -- prediction-only head) but train/collate.py still needs it for aux-label
    supervision, so it must be captured here rather than discarded like the old version did."""
    st = stats if stats is not None else Stats()
    parts = GAP_RE.split(text)
    gaps = GAP_RE.findall(text)
    chars_l, bnd_l, dia_l, cap_l, punct_l, real_l = [], [], [], [], [], []
    for i, seg in enumerate(parts):
        if seg.strip():
            nr = normalize_record(seg, st, with_punct=True)
            if nr is None:
                return None
            c, b, d, cp, p = nr
            chars_l.append(c); bnd_l.append(b); dia_l.append(d); cap_l.append(cp); punct_l.append(p)
            real_l.append(np.zeros(len(c), dtype=bool))
        if i < len(gaps):
            n = len(gaps[i])
            chars_l.append(np.full(n, MASK, np.int64))
            bnd_l.append(np.full(n, UNK_BND, np.int64))
            dia_l.append(np.full(n, UNK_DIA, np.int64))
            cap_l.append(np.zeros(n, np.int64))  # no real cap signal in a gap; never supervised anyway
            punct_l.append(np.full(n, UNK_PUNCT, np.int64))
            real_l.append(np.ones(n, dtype=bool))
    if not chars_l:
        return None
    return (np.concatenate(chars_l), np.concatenate(bnd_l), np.concatenate(dia_l),
            np.concatenate(cap_l), np.concatenate(punct_l), np.concatenate(real_l))


def load_whole_full(split=None, min_len=32, max_len=None, field="with_diacritics", max_records=None):
    """Whole inscriptions, FULL planes (chars/boundary/dia/punct), damage kept in place
    as MASK+unknown positions -- gives the model its natural, native representation
    (real accents/breathings and punctuation, not the Ithaca-mirroring reduced format
    that forces dia/punct to unknown). Ithaca has no capacity to use accents/case/
    punctuation at all -- its own model architecture never sees them, so its natural
    representation IS the deaccented-but-spaced ithaca_text (see text_to_planes());
    ours genuinely can, so this is what "each model's own native representation of
    the same underlying text" means for our side. Both still get word BOUNDARIES:
    Ithaca via literal spaces in its text, ours via its own boundary channel/plane."""
    out = []
    st = Stats()
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        sp = split_of(r.get("PHI_ID"))
        if split and sp != split:
            continue
        text = (r.get(field) or (r.get("ithaca_text") if field == "with_diacritics" else "")) or ""
        if not text:
            continue
        planes = text_to_full_planes(text, st)
        if planes is None:
            continue
        chars, boundary, dia, cap, punct, is_real_lacuna = planes
        if len(chars) < min_len or (max_len and len(chars) > max_len):
            continue
        region = r.get("main_region"); tpq = r.get("tpq"); taq = r.get("taq")
        out.append(dict(chars=chars, boundary=boundary, dia=dia, cap=cap, punct=punct,
                        is_real_lacuna=is_real_lacuna,
                        phi_id=r.get("PHI_ID"), seg=0, split=sp,
                        region=region, tpq=tpq, taq=taq,
                        region_id=region_to_id(region), century_id=record_century_id(tpq, taq)))
        if max_records and len(out) >= max_records:
            return out
    return out


if __name__ == "__main__":
    from collections import Counter
    c, letters = Counter(), Counter()
    for r in load(min_len=32):
        c[r["split"]] += 1
        letters[r["split"]] += len(r["chars"])
    for s in ("train", "val", "test"):
        print(f"{s:5s}: {c[s]:>7,} segments, {letters[s]/1e6:7.1f}M letters")
