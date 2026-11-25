"""Whole-document lacuna handling: text_to_full_planes (iphi.py + papyri.py) must mark real
'-'/'...' damage runs distinctly from synthetically-maskable known text, with the exact known
length for '-' and a random stand-in width for '...' (unknown length)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insc" / "data"))

import iphi
import papyri

MASK, UNK_BND = 24, 3


def test_iphi_known_length_dash_run_marked_real_lacuna():
    text = "λογος--ετερος"  # 2-char real lacuna between two known words
    out = iphi.text_to_full_planes(text)
    assert out is not None
    chars, boundary, dia, cap, punct, is_real = out
    n_lacuna = int(is_real.sum())
    assert n_lacuna == 2, "a '-' run's length is KNOWN -- must match the dash count exactly"
    assert (chars[is_real] == MASK).all()
    assert (boundary[is_real] == UNK_BND).all()
    assert (~is_real).any(), "the surrounding known text must not be marked as lacuna"
    assert len(cap) == len(chars)


def test_iphi_no_lacuna_when_no_dash():
    out = iphi.text_to_full_planes("λογος ετερος")
    assert out is not None
    _, _, _, _, _, is_real = out
    assert not is_real.any()


def test_papyri_dash_run_exact_length():
    rng = np.random.default_rng(0)
    text = "λογος---ετερος"  # 3-char real lacuna, known length
    out = papyri.text_to_full_planes(text, rng)
    assert out is not None
    chars, boundary, dia, cap, punct, is_real = out
    assert int(is_real.sum()) == 3
    assert (chars[is_real] == papyri.MASK).all()
    assert len(cap) == len(chars)


def test_papyri_ellipsis_gets_random_stand_in_width_in_range():
    rng = np.random.default_rng(0)
    widths = []
    for _ in range(30):
        out = papyri.text_to_full_planes("λογος…ετερος", rng)
        assert out is not None
        _, _, _, _, _, is_real = out
        widths.append(int(is_real.sum()))
    assert all(papyri.ELLIPSIS_STAND_IN_MIN <= w <= papyri.ELLIPSIS_STAND_IN_MAX for w in widths)
    assert len(set(widths)) > 1, "stand-in width should vary per draw, not be a fixed constant"


def test_papyri_mixed_dash_and_ellipsis_in_one_text():
    rng = np.random.default_rng(0)
    text = "αλφα--βητα…γαμμα"
    out = papyri.text_to_full_planes(text, rng)
    assert out is not None
    chars, boundary, dia, cap, punct, is_real = out
    # two separate lacuna runs, both real, both MASK
    assert is_real.any()
    assert (chars[is_real] == papyri.MASK).all()
    # some known text survives on both sides
    assert (~is_real).any()


def test_iphi_cap_plane_captured_not_discarded():
    """Regression test: text_to_full_planes used to silently discard the capitalization
    plane (assigned to a throwaway `_cp` variable) -- train/collate.py needs rec["cap"] for
    aux-label supervision, so this must be a real, correctly-sized array."""
    out = iphi.text_to_full_planes("Λογος ετερος")
    chars, boundary, dia, cap, punct, is_real = out
    assert cap.dtype == chars.dtype or cap.shape == chars.shape
    assert cap.sum() > 0, "the capitalized first letter must be reflected in the cap plane"
