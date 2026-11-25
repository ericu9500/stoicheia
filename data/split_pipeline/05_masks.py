#!/usr/bin/env python3
"""Stage 5: per-sentence conflict masks for every record in every tier.

For each sentence of each record, OR together the zone bits of:
  a) its exact skeleton hash in the index
  b) its bag (sorted-words) hash                     [reordered duplicates]
  c) every word-5-gram window over the record's word stream that hits the
     index (the window marks every sentence it overlaps)
  d) document-level MinHash-LSH match (est J >= 0.35) against ANY pristine
     record: the matched record's zone marks ALL sentences (edition-level
     safety net; applied to pristine and repaired records)
Then the record's own zone bit is cleared (a record never conflicts with the
bucket it itself lives in).

Output: work/masks/<tier>/shard_NNNNN.parquet, row-aligned with
work/sentences/<tier>/shard_NNNNN.parquet: (rid, zone int8, masks list<uint16>)
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
from common import h64, NGRAM, ZONE_TRAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PERM = 128
DOC_BANDS, DOC_ROWS = 64, 2
DOC_J = 0.50           # doc-level match threshold (edition-level safety net)
DOC_MIN_WORDS = 35     # both sides must be substantial; short records are
DOC_RATIO = 3.0        # protected by exact/bag sentence matching instead
MERSENNE61 = np.uint64((1 << 61) - 1)
rng = np.random.RandomState(20260709)          # same perms as stage 2
PERM_A = rng.randint(1, 1 << 28, size=N_PERM).astype(np.uint64)
PERM_B = rng.randint(0, 1 << 32, size=N_PERM).astype(np.uint64)
BAND_MIX = np.uint64(0x9E3779B97F4A7C15)

# ---- globals shared with forked workers (read-only, COW) ----
G = {}


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


def setup_globals():
    G["keys"] = np.load(os.path.join(ROOT, "work", "index_keys.npy"), mmap_mode="r")
    G["masks"] = np.load(os.path.join(ROOT, "work", "index_masks.npy"), mmap_mode="r")
    z = np.load(os.path.join(ROOT, "work", "minhash_pristine.npz"))
    rids, zones, sigs = z["rids"], z["zones"].copy(), z["sigs"]
    G["p_nwords"] = z["nwords"]
    t = pq.read_table(os.path.join(ROOT, "work", "zones_final.parquet"),
                      columns=["rid", "zone"])
    zf = dict(zip(t["rid"].to_pylist(), t["zone"].to_pylist()))
    for i, r in enumerate(rids):
        zones[i] = zf.get(str(r), ZONE_TRAIN)
    G["p_zones"] = zones.astype(np.int8)
    G["p_sigs"] = sigs
    ok = zones < ZONE_TRAIN            # only val/test-eligible pristine matters
    idx = np.where(ok)[0]
    bk = band_keys(sigs[idx])          # (m, 64)
    order = np.argsort(bk, axis=0, kind="stable")
    G["lsh_sorted"] = np.take_along_axis(bk, order, axis=0)
    G["lsh_idx"] = idx[order]          # original pristine row per sorted slot
    G["zones_final"] = zf
    print("globals ready: index=%d keys, lsh over %d pristine records"
          % (len(G["keys"]), len(idx)), flush=True)


def lookup(q):
    """q: uint64 array -> uint16 masks (0 where no hit)."""
    keys, masks = G["keys"], G["masks"]
    pos = np.searchsorted(keys, q)
    pos[pos >= len(keys)] = len(keys) - 1
    hit = keys[pos] == q
    out = np.zeros(len(q), dtype=np.uint16)
    out[hit] = masks[pos[hit]]
    return out


def doc_lsh_mask(sig, qwords):
    """Zone bitmask of pristine records with est Jaccard >= DOC_J to sig.

    Guarded against degenerate matches: both sides must have >= DOC_MIN_WORDS
    and sizes within DOC_RATIO of each other (short/boilerplate records are
    protected by the exact/bag sentence layers instead).
    """
    if qwords < DOC_MIN_WORDS:
        return 0
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
        return 0
    cand = np.unique(np.concatenate(cands))
    cw = G["p_nwords"][cand]
    ok = (cw >= DOC_MIN_WORDS) & (cw <= qwords * DOC_RATIO) & \
         (cw * DOC_RATIO >= qwords)
    cand = cand[ok]
    if len(cand) == 0:
        return 0
    est = (G["p_sigs"][cand] == sig[None, :]).mean(axis=1)
    good = cand[est >= DOC_J]
    mask = 0
    for z in np.unique(G["p_zones"][good]):
        mask |= 1 << int(z)
    return mask


def process_shard(args):
    tier, path, out_path = args
    t = pq.read_table(path, columns=["rid", "zone", "skels"])
    rids = t["rid"].to_pylist()
    zones = t["zone"].to_pylist()
    skels_col = t["skels"].to_pylist()
    n = len(rids)

    exact_q, bag_q, gram_q = [], [], []
    exact_loc, gram_loc = [], []       # (rec, sent) / (rec, first_sent, last_sent)
    gram_range = []                    # per record: (start, end) into gram_q
    for i in range(n):
        skels = skels_col[i]
        stream = []
        wsent = []
        for si, sk in enumerate(skels):
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

    sent_masks = [np.zeros(len(skels_col[i]), dtype=np.uint32) for i in range(n)]
    for (i, si), m1, m2 in zip(exact_loc, em, bm):
        if m1 or m2:
            sent_masks[i][si] |= int(m1) | int(m2)
    for (i, s0, s1), m1 in zip(gram_loc, gm):
        if m1:
            sent_masks[i][s0:s1 + 1] |= int(m1)

    # resolve final zones + doc-level LSH + clear own zone
    zf = G["zones_final"]
    final_zones = np.empty(n, dtype=np.int8)
    n_doc_hits = 0
    for i in range(n):
        z = zf.get(rids[i], ZONE_TRAIN)
        final_zones[i] = z
        g0, g1 = gram_range[i]
        if tier in ("pristine", "repaired") and g1 > g0:
            qwords = (g1 - g0) + NGRAM - 1
            dm = doc_lsh_mask(minhash_from_ngram_hashes(gram_arr[g0:g1]), qwords)
            if dm:
                sent_masks[i] |= np.uint32(dm)
                if z >= ZONE_TRAIN or (dm & ~(1 << int(z))):
                    n_doc_hits += 1  # don't count pure self-zone matches
        if z < ZONE_TRAIN:
            sent_masks[i] &= ~np.uint32(1 << int(z))

    out = pa.table({
        "rid": rids,
        "zone": pa.array(final_zones, type=pa.int8()),
        "masks": pa.array([m.astype(np.uint16).tolist() for m in sent_masks],
                          type=pa.list_(pa.uint16())),
    })
    pq.write_table(out, out_path, compression="zstd")
    n_contaminated = sum(1 for m in sent_masks if m.any())
    return n, n_contaminated, n_doc_hits


def main():
    setup_globals()
    tasks = []
    for tier in ("pristine", "repaired", "bronze", "inscriptions"):
        os.makedirs(os.path.join(ROOT, "work", "masks", tier), exist_ok=True)
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            out = os.path.join(ROOT, "work", "masks", tier, os.path.basename(p))
            tasks.append((tier, p, out))
    print("%d shards" % len(tasks), flush=True)
    workers = max(4, os.cpu_count() - 8)
    stats = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (tier, p, _), (nr, nc, nd) in zip(
                tasks, ex.map(process_shard, tasks, chunksize=1)):
            s = stats.setdefault(tier, {"records": 0, "contaminated": 0,
                                        "doc_lsh_hits": 0})
            s["records"] += nr
            s["contaminated"] += nc
            s["doc_lsh_hits"] += nd
    with open(os.path.join(ROOT, "work", "stage5_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
