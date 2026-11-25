"""Documentary papyri (DDbDP via papyri_clean.jsonl) loader — segments, splits, planes.

Mirrors iphi.py's interface exactly (same record dict keys, phi_id := TM number) so the
finetune/eval stack is reusable. Split rule (same as the 10-fold split's fixed
documentary rule): TM last digit 3 -> test, 4 -> val, else train — fold-0's pretraining
excluded TM-3/4 papyri and excised their literary echoes, so both eval splits are
genuinely unseen by the fold-0 torso.

Lacunae: '-' runs (lost chars of known length) AND U+2026 '…' (unknown length); each
record is split at either into segments of continuous known text.
"""
from __future__ import annotations

import json, os, re, sys
from pathlib import Path

import numpy as np

from data.normalize import ALPHABET, Stats, normalize_record  # noqa: F401
try:                    # imported as a package module (python -m insc.data.papyri)
    from insc.data.meta_vocab import UNK_REGION, UNK_CENTURY
except ImportError:     # imported with insc/data on sys.path (how the trainers load it)
    from meta_vocab import UNK_REGION, UNK_CENTURY

JSONL = Path(os.path.expandvars("$AGD_DATA/data/papyri_clean.jsonl"))
GAP_RE = re.compile(r"-+|…+")
_DASH_RE = re.compile(r"-+")
_ELLIPSIS_RE = re.compile(r"…+")

MASK, UNK_BND, UNK_DIA, UNK_PUNCT = 24, 3, 48, 6
ELLIPSIS_STAND_IN_MIN, ELLIPSIS_STAND_IN_MAX = 20, 30  # '…' has no known length -- per-run
                                                        # random stand-in width, never supervised
                                                        # regardless (same as a real '-' run)


def split_of(tm):
    s = str(tm).strip()
    if not s or not s[-1].isdigit():
        return "train"
    test_d = os.environ.get("INSC_TEST_DIGIT", "3")
    val_d = os.environ.get("INSC_VAL_DIGIT", "4")
    return {val_d: "val", test_d: "test"}.get(s[-1], "train")


def text_to_full_planes(text, rng, stats=None):
    """Raw text (with '-' AND '…' damage runs) -> (chars, boundary, dia, punct,
    is_real_lacuna) arrays, WHOLE text, damage kept in place as MASK+unknown positions
    rather than split away. Mirrors iphi.py's text_to_full_planes() exactly, generalized
    to papyri's second lacuna convention: '-' runs have a KNOWN length (count the dashes,
    stonecutter/scribe spacing); '…' runs have an UNKNOWN length -- there's no real count
    to use, so each '…' occurrence gets its own random stand-in width in
    [ELLIPSIS_STAND_IN_MIN, ELLIPSIS_STAND_IN_MAX]. Either way the run is marked
    is_real_lacuna=True and is NEVER a supervision target downstream (no ground truth
    exists for either convention, unlike a synthetically-masked span over known text)."""
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
            g = gaps[i]
            n = len(g) if g[0] == "-" else int(rng.integers(ELLIPSIS_STAND_IN_MIN,
                                                            ELLIPSIS_STAND_IN_MAX + 1))
            chars_l.append(np.full(n, MASK, np.int64))
            bnd_l.append(np.full(n, UNK_BND, np.int64))
            dia_l.append(np.full(n, UNK_DIA, np.int64))
            cap_l.append(np.zeros(n, np.int64))
            punct_l.append(np.full(n, UNK_PUNCT, np.int64))
            real_l.append(np.ones(n, dtype=bool))
    if not chars_l:
        return None
    return (np.concatenate(chars_l), np.concatenate(bnd_l), np.concatenate(dia_l),
            np.concatenate(cap_l), np.concatenate(punct_l), np.concatenate(real_l))


def load_whole_full(split=None, min_len=32, max_len=None, field="text", max_records=None,
                    seed=0):
    """Whole papyri, FULL planes (chars/boundary/dia/punct) + is_real_lacuna, damage kept
    in place as MASK+unknown positions instead of split away -- mirrors iphi.py's
    load_whole_full(). See text_to_full_planes() for the '-' vs '…' handling."""
    out = []
    st = Stats()
    rng = np.random.default_rng(seed)
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        sp = split_of(r.get("TM"))
        if split and sp != split:
            continue
        text = r.get(field) or ""
        if not text:
            continue
        planes = text_to_full_planes(text, rng, st)
        if planes is None:
            continue
        chars, boundary, dia, cap, punct, is_real_lacuna = planes
        if len(chars) < min_len or (max_len and len(chars) > max_len):
            continue
        out.append(dict(chars=chars, boundary=boundary, dia=dia, cap=cap, punct=punct,
                        is_real_lacuna=is_real_lacuna,
                        phi_id=r.get("TM"), seg=0, split=sp,
                        region=None, tpq=None, taq=None,
                        region_id=UNK_REGION, century_id=UNK_CENTURY))
        if max_records and len(out) >= max_records:
            return out
    return out


def load(split=None, min_len=32, field="text", max_records=None):
    """Yield dicts: chars/boundary/dia/cap/punct planes + phi_id (=TM)/seg/split.
    One dict per continuous SEGMENT (split at '-' runs and '…')."""
    out = []
    st = Stats()
    for line in JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        sp = split_of(r.get("TM"))
        if split and sp != split:
            continue
        text = r.get(field) or ""
        for seg_i, seg in enumerate(GAP_RE.split(text)):
            if len(seg.strip()) < min_len:
                continue
            nr = normalize_record(seg, st, with_punct=True)
            if nr is None or len(nr[0]) < min_len:
                continue
            chars, boundary, dia, cap, punct = nr
            out.append(dict(
                chars=chars, boundary=boundary, dia=dia, cap=cap, punct=punct,
                phi_id=r.get("TM"), seg=seg_i, split=sp,
                region=None, tpq=None, taq=None,
                # papyri_clean.jsonl carries no date/place metadata (TM/file/text only) --
                # always UNK. Papyrus records therefore ride the region/century embedding
                # at its "unknown" row; only I.PHI (iphi.py) records are actually primed.
                region_id=UNK_REGION, century_id=UNK_CENTURY))
            if max_records and len(out) >= max_records:
                return out
    return out
