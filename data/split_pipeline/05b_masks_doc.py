#!/usr/bin/env python3
"""Stage 5b (documentary-clean variant): per-sentence documentary-hit masks.

Matching layers against the stage-4b documentary index, for the three trainable
tiers (pristine, repaired, bronze):
  a) exact skeleton hash
  b) bag (sorted-words) hash
  c) word-8-gram windows, marking every sentence they overlap
  d) document-level MinHash-LSH match (est J >= DOC_J) against the stage-4c
     documentary signature index -- catches whole-RECORD near-duplicates that
     (a)-(c) can miss: scanned SOURCEBOOK volumes (Dittenberger's Sylloge,
     Schwyzer's Dialectorum Graecarum Exempla Epigraphica Potiora, Cagnat's
     Inscriptiones Graecae ad Res Romanas Pertinentes, and similar epigraphic/
     papyrological corpora catalogued as ordinary "literary" books) reproduce
     documentary text with enough OCR noise / editorial apparatus / formatting
     drift that individual 8-grams can slip through while the document is still
     substantially the same content. A doc-level match flags EVERY sentence of
     the record, so 06b_doc_clean.py drops the whole record (not just the
     overlapping span) -- the same "wholesale over partial" policy already used
     for Greek-origin bronze.

Output: work/doc_clean/masks/<tier>/shard_NNNNN.parquet, row-aligned with
work/sentences/<tier>/shard_NNNNN.parquet: (rid, masks list<uint16>).
A nonzero mask entry means "this sentence textually collides with some
documentary text (any PHI/TM digit)".
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, NGRAM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "doc_clean")
G = {}

MASK_DOC = np.uint16(1 << 13)   # same bit stage 4b/4c index entries carry
N_PERM = 128
DOC_BANDS, DOC_ROWS = 64, 2
DOC_J = 0.50            # doc-level match threshold (same as stage 5's edition-level net)
DOC_MIN_WORDS = 35       # both sides must be substantial; short records are
DOC_RATIO = 3.0          # protected by exact/bag sentence matching instead
MERSENNE61 = np.uint64((1 << 61) - 1)
rng = np.random.RandomState(20260709)          # same perms as stage 2 / 4c / 5
PERM_A = rng.randint(1, 1 << 28, size=N_PERM).astype(np.uint64)
PERM_B = rng.randint(0, 1 << 32, size=N_PERM).astype(np.uint64)
BAND_MIX = np.uint64(0x9E3779B97F4A7C15)


def setup_globals():
    G["keys"] = np.load(os.path.join(OUT, "index_keys.npy"), mmap_mode="r")
    G["masks"] = np.load(os.path.join(OUT, "index_masks.npy"), mmap_mode="r")
    print("doc index: %d keys" % len(G["keys"]), flush=True)

    z = np.load(os.path.join(OUT, "minhash_documentary.npz"))
    G["d_nwords"] = z["nwords"]
    sigs = z["sigs"]
    bk = band_keys(sigs)                # (m, 64)
    order = np.argsort(bk, axis=0, kind="stable")
    G["lsh_sorted"] = np.take_along_axis(bk, order, axis=0)
    G["lsh_idx"] = order                # original documentary row per sorted slot
    G["lsh_sigs"] = sigs
    print("doc minhash: %d documentary records indexed for LSH" % len(sigs), flush=True)


def lookup(q):
    keys, masks = G["keys"], G["masks"]
    pos = np.searchsorted(keys, q)
    pos[pos >= len(keys)] = len(keys) - 1
    hit = keys[pos] == q
    out = np.zeros(len(q), dtype=np.uint16)
    out[hit] = masks[pos[hit]]
    return out


def minhash_from_ngram_hashes(hs):
    if len(hs) == 0:
        return np.zeros(N_PERM, dtype=np.uint64)
    h = np.asarray(hs, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    v = (PERM_A[:, None] * h[None, :] + PERM_B[:, None]) % MERSENNE61
    return v.min(axis=1)


def band_keys(sig):
    """sig (..,128) -> (..,64) uint64 band keys."""
    s = sig.reshape(sig.shape[:-1] + (DOC_BANDS, DOC_ROWS))
    return (s[..., 0] * BAND_MIX + s[..., 1])


def doc_lsh_hit(sig, qwords):
    """True if some documentary record has est Jaccard >= DOC_J to sig.

    Guarded against degenerate matches: both sides must have >= DOC_MIN_WORDS
    and sizes within DOC_RATIO of each other (short/boilerplate records are
    protected by the exact/bag/8-gram sentence layers instead).
    """
    if qwords < DOC_MIN_WORDS:
        return False
    bk = band_keys(sig[None, :])[0]    # (64,)
    srt, sidx = G["lsh_sorted"], G["lsh_idx"]
    cands = []
    for b in range(DOC_BANDS):
        col = srt[:, b]
        lo = np.searchsorted(col, bk[b], side="left")
        hi = np.searchsorted(col, bk[b], side="right")
        if hi > lo:
            cands.append(sidx[lo:hi, b])
    if not cands:
        return False
    cand = np.unique(np.concatenate(cands))
    cw = G["d_nwords"][cand]
    ok = (cw >= DOC_MIN_WORDS) & (cw <= qwords * DOC_RATIO) & \
         (cw * DOC_RATIO >= qwords)
    cand = cand[ok]
    if len(cand) == 0:
        return False
    est = (G["lsh_sigs"][cand] == sig[None, :]).mean(axis=1)
    return bool((est >= DOC_J).any())


def process_shard(args):
    tier, path, out_path = args
    t = pq.read_table(path, columns=["rid", "skels"])
    rids = t["rid"].to_pylist()
    skels_col = t["skels"].to_pylist()
    n = len(rids)

    exact_q, bag_q, gram_q = [], [], []
    exact_loc, gram_loc = [], []
    gram_range = []                    # per record: (start, end) into gram_q
    for i in range(n):
        stream, wsent = [], []
        for si, sk in enumerate(skels_col[i]):
            w = sk.split()
            exact_q.append(h64(sk))
            bag_q.append(h64(" ".join(sorted(w))))
            exact_loc.append((i, si))
            stream.extend(w)
            wsent.extend([si] * len(w))
        g0 = len(gram_q)
        if len(stream) >= NGRAM:
            for j in range(len(stream) - NGRAM + 1):
                gram_q.append(h64(" ".join(stream[j:j + NGRAM])))
                gram_loc.append((i, wsent[j], wsent[j + NGRAM - 1]))
        gram_range.append((g0, len(gram_q)))

    gram_arr = np.array(gram_q, dtype=np.uint64)
    em = lookup(np.array(exact_q, dtype=np.uint64)) if exact_q else np.zeros(0, np.uint16)
    bm = lookup(np.array(bag_q, dtype=np.uint64)) if bag_q else np.zeros(0, np.uint16)
    gm = lookup(gram_arr) if gram_q else np.zeros(0, np.uint16)

    sent_masks = [np.zeros(len(skels_col[i]), dtype=np.uint16) for i in range(n)]
    for (i, si), m1, m2 in zip(exact_loc, em, bm):
        if m1 or m2:
            sent_masks[i][si] |= m1 | m2
    for (i, s0, s1), m1 in zip(gram_loc, gm):
        if m1:
            sent_masks[i][s0:s1 + 1] |= m1

    n_doc_hits = 0
    for i in range(n):
        g0, g1 = gram_range[i]
        if g1 <= g0:
            continue
        qwords = (g1 - g0) + NGRAM - 1
        if doc_lsh_hit(minhash_from_ngram_hashes(gram_arr[g0:g1]), qwords):
            sent_masks[i][:] |= MASK_DOC     # whole-record echo: flag every sentence
            n_doc_hits += 1

    out = pa.table({
        "rid": rids,
        "masks": pa.array([m.tolist() for m in sent_masks],
                          type=pa.list_(pa.uint16())),
    })
    pq.write_table(out, out_path, compression="zstd")
    n_contaminated = sum(1 for m in sent_masks if m.any())
    return n, n_contaminated, n_doc_hits


def main():
    setup_globals()
    tasks = []
    for tier in ("pristine", "repaired", "bronze"):
        os.makedirs(os.path.join(OUT, "masks", tier), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            out = os.path.join(OUT, "masks", tier, os.path.basename(p))
            tasks.append((tier, p, out))
    print("%d shards" % len(tasks), flush=True)
    workers = max(4, min(16, (os.cpu_count() or 12) - 8))
    stats = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (tier, p, _), (nr, nc, nd) in zip(
                tasks, ex.map(process_shard, tasks, chunksize=1)):
            s = stats.setdefault(tier, {"records": 0, "contaminated": 0,
                                        "doc_lsh_hits": 0})
            s["records"] += nr
            s["contaminated"] += nc
            s["doc_lsh_hits"] += nd
    with open(os.path.join(OUT, "stage5b_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
