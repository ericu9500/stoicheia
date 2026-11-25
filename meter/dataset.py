"""Annotated lines -> letter-plane records -> packed model batches.

A record is one annotated unit (a macronized line, or a window of consecutive
verses) encoded through the backbone codec with per-letter label planes:

  y_mac   -100 everywhere except marked dichrona (0 long / 1 short)
  y_scan  -100 for macron-only records; else 0 none / 1 heavy / 2 light / 3 verse
          at every letter (the "no syllable ends here" class is supervised too)

Records are greedily packed into fixed-T rows with per-record seg_ids, exactly like
pretraining's document packing (block-diagonal attention). All input planes carry
their true values — they are all known from raw text at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from meter.backbone import PAD_ID, Stats, normalize_record
from meter.marks import IGNORE, SCAN_O, parse_macron_line, parse_scan_line

# sentence-final punctuation classes in the codec (period, question); a verse whose
# text does not end with one of these ends on a word boundary, not a sentence one
_SENT_PUNCT = (4, 5)


@dataclass
class Record:
    chars: np.ndarray     # uint8 letter ids
    boundary: np.ndarray  # uint8 0/1/2
    dia: np.ndarray       # uint8 packed diacritics
    punct: np.ndarray     # uint8 punct class
    cap: np.ndarray       # uint8 0/1
    y_mac: np.ndarray     # int8  -100/0/1
    y_scan: np.ndarray    # int8  -100 or 0..3

    def __len__(self):
        return len(self.chars)


def _encode_plain(plain: str) -> tuple | None:
    r = normalize_record(plain, Stats(), with_punct=True)
    if r is None:
        return None
    return r  # chars, boundary, dia, cap, punct


def encode_plain(plain: str) -> Record | None:
    """Unlabeled text (inference input)."""
    r = _encode_plain(plain)
    if r is None:
        return None
    chars, boundary, dia, cap, punct = r
    n = len(chars)
    return Record(chars, boundary, dia, punct, cap,
                  np.full(n, IGNORE, np.int8), np.full(n, IGNORE, np.int8))


def _with_labels(plain: str, labels: dict[int, int], task: str) -> Record | None:
    r = _encode_plain(plain)
    if r is None:
        return None
    chars, boundary, dia, cap, punct = r
    n = len(chars)
    if labels and max(labels) >= n:
        return None  # letter-count mismatch between annotation walk and codec
    y_mac = np.full(n, IGNORE, np.int8)
    y_scan = np.full(n, IGNORE, np.int8)
    if task == "mac":
        for k, v in labels.items():
            y_mac[k] = v
    else:
        y_scan[:] = SCAN_O
        for k, v in labels.items():
            y_scan[k] = v
    return Record(chars, boundary, dia, punct, cap, y_mac, y_scan)


def encode_macron_line(marked: str) -> Record | None:
    plain, labels = parse_macron_line(marked)
    if not labels:
        return None
    return _with_labels(plain, labels, "mac")


def encode_scan_line(bracketed: str) -> Record | None:
    parsed = parse_scan_line(bracketed)
    if parsed is None:
        return None
    plain, labels = parsed
    return _with_labels(plain, labels, "scan")


def concat_verses(recs: list[Record]) -> Record:
    """Join consecutive verse records into one stream record.

    The codec stamps boundary=2 (sentence end) on every record's last letter; inside
    a window that would leak verse segmentation to the input, so seams are demoted to
    word boundaries unless the verse really ends with sentence punctuation. The
    window-final letter keeps 2 (a record end, exactly as in pretraining packing).
    """
    bnd = [r.boundary.copy() for r in recs]
    for i, r in enumerate(recs[:-1]):
        if r.punct[-1] not in _SENT_PUNCT:
            bnd[i][-1] = 1
    return Record(*(np.concatenate(x) for x in (
        [r.chars for r in recs], bnd, [r.dia for r in recs],
        [r.punct for r in recs], [r.cap for r in recs],
        [r.y_mac for r in recs], [r.y_scan for r in recs])))


def make_windows(verses: list[Record], rng: np.random.Generator,
                 passes: int, max_verses: int, T: int) -> list[Record]:
    """Random runs of 1..max_verses consecutive verses from one work, each pass
    starting at a fresh offset, every window capped at T letters."""
    out = []
    nv = len(verses)
    for _ in range(passes):
        i = int(rng.integers(0, min(max_verses, nv)))
        while i < nv:
            k = int(rng.integers(1, max_verses + 1))
            group, total = [], 0
            for r in verses[i:i + k]:
                if total + len(r) > T:
                    break
                group.append(r)
                total += len(r)
            if group:
                out.append(concat_verses(group) if len(group) > 1 else group[0])
                i += len(group)
            else:
                i += 1  # single verse longer than T: skip it
    return out


# ---------------------------------------------------------------- packing

def pack_records(records, T=2048, order=None):
    """Greedy packing of whole records into rows of <= T letters. Returns
    (rows, n_skipped) where each row is a list of record indices."""
    order = range(len(records)) if order is None else order
    rows, skipped = [], 0
    cur, cur_n = [], 0
    for ri in order:
        r = records[ri]
        if r is None:
            continue
        n = len(r)
        if n > T:
            skipped += 1
            continue
        if cur_n + n > T:
            rows.append(cur)
            cur, cur_n = [], 0
        cur.append(ri)
        cur_n += n
    if cur:
        rows.append(cur)
    return rows, skipped


def batch_rows(rows, records, T=2048, device=None, with_slots=False):
    """Stack rows (lists of record indices) into model + label tensors.

    Returns input_ids/boundary/dia/punct/cap/seg_id (B,T) plus y_mac/y_scan (B,T);
    with_slots also returns per-row [(record_index, char_offset)] for mapping
    per-position predictions back to records.
    """
    B = len(rows)
    ids = np.full((B, T), PAD_ID, dtype=np.int64)
    bnd = np.zeros((B, T), dtype=np.int64)
    dia = np.zeros((B, T), dtype=np.int64)
    pct = np.zeros((B, T), dtype=np.int64)
    cp = np.zeros((B, T), dtype=np.int64)
    seg = np.zeros((B, T), dtype=np.int64)
    y_m = np.full((B, T), IGNORE, dtype=np.int64)
    y_s = np.full((B, T), IGNORE, dtype=np.int64)
    slots = []
    for b, row in enumerate(rows):
        c = 0
        rs = []
        for k, ri in enumerate(row):
            r = records[ri]
            n = len(r)
            ids[b, c:c + n] = r.chars
            bnd[b, c:c + n] = r.boundary
            dia[b, c:c + n] = r.dia
            pct[b, c:c + n] = r.punct
            cp[b, c:c + n] = r.cap
            seg[b, c:c + n] = k + 1
            y_m[b, c:c + n] = r.y_mac
            y_s[b, c:c + n] = r.y_scan
            rs.append((ri, c))
            c += n
        slots.append(rs)
    t = lambda a: torch.from_numpy(a) if device is None else torch.from_numpy(a).to(device)
    out = dict(input_ids=t(ids), boundary=t(bnd), dia=t(dia), punct=t(pct),
               cap=t(cp), seg_id=t(seg), y_mac=t(y_m), y_scan=t(y_s))
    if with_slots:
        out["slots"] = slots
    return out


# ---------------------------------------------------------------- npz store

_FIELDS = ("chars", "boundary", "dia", "punct", "cap", "y_mac", "y_scan")


def save_records(path, records, works=None):
    """Concatenate records into one npz (offsets + planes [+ per-record work names])."""
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in records], out=offsets[1:])
    arrays = {f: np.concatenate([getattr(r, f) for r in records]) if records
              else np.zeros(0, np.int8) for f in _FIELDS}
    extra = {}
    if works is not None:
        extra["works"] = np.array(works)
    np.savez_compressed(path, offsets=offsets, **arrays, **extra)


def load_records(path):
    """-> (records, works|None)"""
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    planes = {f: z[f] for f in _FIELDS}
    records = [Record(**{f: planes[f][off[i]:off[i + 1]] for f in _FIELDS})
               for i in range(len(off) - 1)]
    works = [str(w) for w in z["works"]] if "works" in z else None
    return records, works
