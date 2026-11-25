"""Property tests for data/normalize.py.

The load-bearing invariant: normalize() must be exactly invertible to
(a) the accentless spaced/sentence-marked text and (b) the polytonic word list,
where the reference for both is computed by an INDEPENDENT, slow, obviously-correct
pure-Python implementation. Any misalignment between chars and label planes breaks these.
"""
import sys, unicodedata
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import (ALPHABET, LETTER_IDS, _EXTRA_BASE, ARCHAIC, DIA_STATES,
                            Stats, denormalize, normalize_record, restore_polytonic,
                            unpack_dia, _pack_dia)

import os
RAW = Path(os.path.expandvars(os.environ.get("STOICHEIA_DATA", "$STOICHEIA_DATA"))) \
    / "raw/AncientGreek/data"


# ------------------------------------------------------------ reference impl

def _ref_letter(ch):
    low = ch.lower()
    if low in LETTER_IDS:
        return low
    if low in _EXTRA_BASE:
        return _EXTRA_BASE[low]
    return None


def ref_words_accentless(text):
    """Independent reference: (accentless word, sentence_final) pairs."""
    nfd = unicodedata.normalize("NFD", text)
    words, cur, seen_spunct = [], [], False
    sfinal = {".", ";", "!", "?"}
    for ch in nfd:
        if ch in ARCHAIC or unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            if ch in ARCHAIC and cur:
                pass  # archaic letters act as separators (weight 1) in the kernel
            if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
                continue
        letter = _ref_letter(ch)
        if letter is not None:
            cur.append(letter)
            seen_spunct = False
        else:
            if cur:
                words.append(["".join(cur), ch in sfinal])
                cur = []
            elif words and ch in sfinal:
                words[-1][1] = True
            continue
    if cur:
        words.append(["".join(cur), True])
    if words:
        words[-1][1] = True
    return [(w, s) for w, s in words]


def ref_polytonic_words(text):
    """Reference polytonic words: NFC-fold each maximal run of Greek letters+marks."""
    nfd = unicodedata.normalize("NFD", text)
    keep_marks = {0x0301, 0x0341, 0x0300, 0x0340, 0x0342, 0x0302, 0x0313, 0x0343,
                  0x0314, 0x0345, 0x0308}
    words, cur = [], []
    for ch in nfd:
        if _ref_letter(ch) is not None:
            cur.append(ch)
        elif unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            if cur and ord(ch) in keep_marks:
                cur.append(ch)
        else:
            if cur:
                words.append(unicodedata.normalize("NFC", "".join(cur)))
                cur = []
    if cur:
        words.append(unicodedata.normalize("NFC", "".join(cur)))
    return words


def _fold(w):
    """Fold a reference word the way the pipeline folds letters (case, variants, ς)."""
    out = []
    nfd = unicodedata.normalize("NFD", w)
    for ch in nfd:
        if unicodedata.category(ch) in ("Mn", "Mc", "Me"):
            out.append(ch)
        else:
            l = _ref_letter(ch)
            cap = ch != ch.lower() or ch == "Ϲ"
            out.append(l.upper() if cap else l)
    s = "".join(out)
    return unicodedata.normalize("NFC", s)


def _final_sigma(w):
    # base string uses σ internally; word-final becomes ς — mirror restore_polytonic
    nfc = unicodedata.normalize("NFC", w)
    if nfc.endswith("σ"):
        return nfc[:-1] + "ς"
    return nfc


# ------------------------------------------------------------ unit cases

def norm(text):
    st = Stats()
    r = normalize_record(text, st)
    assert r is not None
    return (*r, st)


def test_kai_ho_logos():
    chars, boundary, dia, cap, st = norm("καὶ ὁ λόγος.")
    assert "".join(ALPHABET[c] for c in chars) == "καιολογοσ"
    assert boundary.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 2]
    accs = [unpack_dia(int(d))[0] for d in dia]
    brs = [unpack_dia(int(d))[1] for d in dia]
    assert accs == [0, 0, 2, 0, 0, 1, 0, 0, 0]      # grave on ι, acute on ο of λόγος
    assert brs == [0, 0, 0, 2, 0, 0, 0, 0, 0]       # rough on ὁ
    assert cap.tolist() == [0] * 9
    assert st.words == 3 and st.sentences == 1


def test_punct_without_space_is_sentence_final():
    chars, boundary, dia, cap, _ = norm("σπείρεις.Πλίνθον πλύνεις")
    text = denormalize(chars, boundary)
    assert text.startswith("σπειρεισ. πλινθον πλυνεισ")
    assert cap[8] == 1  # Π


def test_elision_apostrophe_is_word_boundary():
    for apo in ["᾽", "'", "’", "᾽"]:
        chars, boundary, dia, cap, _ = norm(f"ἀλλ{apo} οὐ γάρ")
        assert denormalize(chars, boundary) == "αλλ ου γαρ."


def test_capital_with_breathing():
    chars, boundary, dia, cap, _ = norm("Ὅλον")
    assert cap[0] == 1
    acc, br, io, dd = unpack_dia(int(dia[0]))
    assert br == 2 and acc == 1                      # rough breathing + acute


def test_iota_subscript_and_diaeresis():
    chars, boundary, dia, cap, _ = norm("τῷ Ῥόδῳ ἀΐδιος")
    a0 = unpack_dia(int(dia[1]))                     # ῷ: circumflex + iota-sub
    assert a0[0] == 3 and a0[2] == 1
    r = unpack_dia(int(dia[2]))                      # Ῥ rough
    assert r[1] == 2 and cap[2] == 1
    ii = unpack_dia(int(dia[7]))                     # ΐ of ἀΐδιος: acute + diaeresis
    assert ii[0] == 1 and ii[3] == 1


def test_final_and_lunate_sigma_merge():
    chars, _, _, cap, _ = norm("λόγος λόγοϲ ΛΟΓΟΣ")
    s = "".join(ALPHABET[c] for c in chars)
    assert s == "λογοσ" * 3
    assert cap[10:15].tolist() == [1] * 5


def test_dia_pack_unpack_roundtrip():
    for acc in range(4):
        for br in range(3):
            for io in range(2):
                for dd in range(2):
                    d = _pack_dia(acc, br, io, dd)
                    assert 0 <= d < DIA_STATES
                    assert unpack_dia(int(d)) == (acc, br, io, dd)


def test_drops_nongreek_record():
    st = Stats()
    assert normalize_record("This is an English sentence entirely.", st) is None
    assert st.records_dropped_nongreek + st.records_dropped_empty == 1


def test_restore_polytonic_simple():
    chars, boundary, dia, cap, _ = norm("καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.")
    words = restore_polytonic(chars, dia, cap, boundary)
    assert words == ["καὶ", "ὁ", "λόγος", "ἦν", "πρὸς", "τὸν", "θεόν"]


# ------------------------------------------------------------ corpus property test

@pytest.mark.parametrize("shard", ["pristine/pristine-00000.parquet",
                                   "repaired/repaired-00000.parquet"])
def test_roundtrip_on_real_records(shard):
    pq = pytest.importorskip("pyarrow.parquet")
    if not (RAW / shard).is_file():
        pytest.skip("pretraining corpus shards not present under $STOICHEIA_DATA")
    t = pq.read_table(RAW / shard, columns=["text"])
    rng = np.random.default_rng(0)
    idx = rng.choice(t.num_rows, size=200, replace=False)
    texts = [t.column("text")[int(i)].as_py() for i in idx]
    st = Stats()
    checked = 0
    for text in texts:
        r = normalize_record(text, st)
        if r is None:
            continue
        chars, boundary, dia, cap = r
        assert len(chars) == len(boundary) == len(dia) == len(cap)

        # (a) accentless words + sentence flags match the reference exactly
        ref = ref_words_accentless(text)
        got_words, got_flags, cur = [], [], []
        for c, b in zip(chars, boundary):
            cur.append(ALPHABET[c])
            if b >= 1:
                got_words.append("".join(cur)); got_flags.append(b == 2); cur = []
        assert got_words == [w for w, _ in ref], f"word mismatch in record"
        assert got_flags == [s for _, s in ref], f"sentence-flag mismatch"

        # (b) polytonic restoration matches the reference NFC words
        ref_poly = [_final_sigma(_fold(w)) for w in ref_polytonic_words(text)]
        got_poly = restore_polytonic(chars, dia, cap, boundary)
        assert got_poly == ref_poly, "polytonic mismatch"
        checked += 1
    assert checked >= 150  # most sampled records survive filtering
