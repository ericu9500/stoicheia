#!/usr/bin/env python3
"""Stage 2: duplicate-aware clustering of literary pristine records.

Groups near-identical EDITIONS of the same text so they share a rotating
bucket. Deliberately tight: transitive closure over loose content overlap
(quotes, psalms inside anthologies, liturgical boilerplate) previously merged
97% of records into one blob. Loose cross-bucket overlap is NOT a leakage
problem -- stage 5's sentence/5-gram/doc-LSH masks excise it from train -- it
only costs data, so clustering optimizes bucket balance, not recall.

Evidence for union (page/record level; volumes pre-grouped by id prefix):
  1. work-prefix of the id (volume/work granularity per source)
  2. identical full-record skeleton hash
  3. MinHash-LSH candidate pairs verified at est Jaccard >= VERIFY_J (0.70)
  4. shared >=5-word sentence skeletons covering >= 50% of the shorter
     record's sentences (and >= 2 shared), shorter side >= 2 sentences

Outputs:
  work/minhash_pristine.npz  (rids, zones, sigs uint64 [n,128], nwords)
  work/clusters.parquet      (rid, cluster, nwords) for literary pristine
  work/stage2_stats.json
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, ngram_hashes, work_prefix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT = os.path.join(ROOT, "work", "sentences", "pristine")

N_PERM = 128
LSH_BANDS, LSH_ROWS = 32, 4          # candidate threshold ~ 0.42
VERIFY_J = float(os.environ.get("VERIFY_J", "0.70"))
LSH_BUCKET_CAP = 200                 # verify pairwise up to this bucket size
SENT_MIN_WORDS = 5
SENT_HASH_CAP = 24
PAIR_MIN_SHARED = 2
PAIR_MIN_FRAC = float(os.environ.get("PAIR_MIN_FRAC", "0.5"))
MERSENNE61 = np.uint64((1 << 61) - 1)

rng = np.random.RandomState(20260709)
PERM_A = rng.randint(1, 1 << 28, size=N_PERM).astype(np.uint64)
PERM_B = rng.randint(0, 1 << 32, size=N_PERM).astype(np.uint64)


def minhash(shingle_hashes):
    if not shingle_hashes:
        return np.zeros(N_PERM, dtype=np.uint64)
    h = np.asarray(shingle_hashes, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    v = (PERM_A[:, None] * h[None, :] + PERM_B[:, None]) % MERSENNE61
    return v.min(axis=1)


def process_shard(path):
    t = pq.read_table(path, columns=["rid", "source", "zone", "skels"])
    rids = t["rid"].to_pylist()
    sources = t["source"].to_pylist()
    zones = t["zone"].to_pylist()
    skels_col = t["skels"].to_pylist()
    n = len(rids)
    sigs = np.zeros((n, N_PERM), dtype=np.uint64)
    rec_hash = np.zeros(n, dtype=np.uint64)
    nwords = np.zeros(n, dtype=np.int32)
    nsents = np.zeros(n, dtype=np.int32)
    prefixes = []
    sent_entries = []  # (sent_hash, row_idx) for >=5-word sentences, literary only
    for i in range(n):
        skels = skels_col[i]
        words = []
        for sk in skels:
            words.extend(sk.split())
        nwords[i] = len(words)
        nsents[i] = len(skels)
        rec_hash[i] = h64(" ".join(words))
        sigs[i] = minhash(ngram_hashes(words))
        prefixes.append(work_prefix(sources[i], rids[i]))
        if zones[i] == -1:
            for sk in skels:
                if sk.count(" ") >= SENT_MIN_WORDS - 1:
                    sent_entries.append((h64(sk), i))
    return (path, rids, np.array(zones, dtype=np.int8), sigs, rec_hash,
            nwords, nsents, prefixes, sent_entries)


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def top_components(uf, nodes, nwords, label, stats):
    sizes = Counter()
    words = Counter()
    for i in nodes:
        r = uf.find(i)
        sizes[r] += 1
        words[r] += int(nwords[i])
    top = words.most_common(5)
    stats[label] = {"n_components": len(sizes),
                    "top5_words": [int(w) for _, w in top],
                    "top5_sizes": [int(sizes[r]) for r, _ in top]}
    print("%s: %d components, top5 words %s" %
          (label, len(sizes), [int(w) for _, w in top]), flush=True)


def main():
    paths = sorted(glob.glob(os.path.join(SENT, "shard_*.parquet")))
    print("%d pristine shards" % len(paths), flush=True)
    workers = max(4, min(64, os.cpu_count() - 8))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(process_shard, paths, chunksize=1):
            results.append(r)
    results.sort(key=lambda r: r[0])

    all_rids, all_zones, all_sigs, all_rech, all_nw, all_ns, all_pref = \
        [], [], [], [], [], [], []
    sent_map = defaultdict(list)      # sent hash -> global row idxs (literary)
    for (_, rids, zones, sigs, rech, nw, ns, prefs, sents) in results:
        base = len(all_rids)
        all_rids.extend(rids)
        all_zones.append(zones)
        all_sigs.append(sigs)
        all_rech.append(rech)
        all_nw.append(nw)
        all_ns.append(ns)
        all_pref.extend(prefs)
        for h, i in sents:
            sent_map[h].append(base + i)
    zones = np.concatenate(all_zones)
    sigs = np.vstack(all_sigs)
    rech = np.concatenate(all_rech)
    nwords = np.concatenate(all_nw)
    nsents = np.concatenate(all_ns)
    n = len(all_rids)
    lit = np.where(zones == -1)[0]
    print("pristine records: %d (literary: %d)" % (n, len(lit)), flush=True)

    np.savez(os.path.join(ROOT, "work", "minhash_pristine.npz"),
             rids=np.array(all_rids), zones=zones, sigs=sigs, nwords=nwords)

    uf = UF(n)
    stats = {"verify_j": VERIFY_J, "pair_min_frac": PAIR_MIN_FRAC}

    # 1. work-prefix
    by_pref = defaultdict(list)
    for i in lit:
        by_pref[all_pref[i]].append(i)
    for grp in by_pref.values():
        for j in grp[1:]:
            uf.union(grp[0], j)
    stats["prefix_groups"] = len(by_pref)
    top_components(uf, lit, nwords, "after_prefix", stats)

    # 2. identical record skeleton
    by_h = defaultdict(list)
    for i in lit:
        by_h[rech[i]].append(i)
    ne = 0
    for grp in by_h.values():
        for j in grp[1:]:
            uf.union(grp[0], j)
            ne += 1
    stats["exact_dup_edges"] = ne
    top_components(uf, lit, nwords, "after_exact", stats)

    # 3. MinHash LSH candidates, ALL verified at est J >= VERIFY_J
    ne = 0
    skipped_hot_bands = 0
    lit_sigs = sigs[lit]
    for b in range(LSH_BANDS):
        band = np.ascontiguousarray(lit_sigs[:, b * LSH_ROWS:(b + 1) * LSH_ROWS])
        bh = band.view(np.dtype((np.void, band.dtype.itemsize * LSH_ROWS))).ravel()
        buckets = defaultdict(list)
        for k, key in enumerate(bh):
            buckets[key.tobytes()].append(k)
        for grp in buckets.values():
            if len(grp) < 2:
                continue
            if len(grp) > LSH_BUCKET_CAP:
                skipped_hot_bands += 1
                continue
            for x in range(len(grp)):
                for y in range(x + 1, len(grp)):
                    a, c = lit[grp[x]], lit[grp[y]]
                    if uf.find(a) == uf.find(c):
                        continue
                    est = float(np.mean(sigs[a] == sigs[c]))
                    if est >= VERIFY_J:
                        uf.union(a, c)
                        ne += 1
    stats["lsh_edges"] = ne
    stats["lsh_hot_bands_skipped"] = skipped_hot_bands
    top_components(uf, lit, nwords, "after_lsh", stats)

    # 4. shared-sentence pair counting (>=50% of the shorter record)
    pair_counts = Counter()
    skipped_hot = 0
    for h, idxs in sent_map.items():
        if len(idxs) < 2:
            continue
        if len(idxs) > SENT_HASH_CAP:
            skipped_hot += 1
            continue
        u = sorted(set(idxs))
        for x in range(len(u)):
            for y in range(x + 1, len(u)):
                pair_counts[(u[x], u[y])] += 1
    ne = 0
    for (a, b), c in pair_counts.items():
        mn = min(nsents[a], nsents[b])
        if mn >= 2 and c >= PAIR_MIN_SHARED and c >= PAIR_MIN_FRAC * mn:
            if uf.find(a) != uf.find(b):
                uf.union(a, b)
                ne += 1
    stats["sentence_edges"] = ne
    stats["hot_sentence_hashes_skipped"] = skipped_hot
    top_components(uf, lit, nwords, "after_sentences", stats)

    # components
    comp = {}
    cluster_of = np.full(n, -1, dtype=np.int64)
    for i in lit:
        r = uf.find(i)
        comp.setdefault(r, len(comp))
        cluster_of[i] = comp[r]
    sizes = Counter(cluster_of[lit])
    top = sizes.most_common(10)
    stats["n_clusters"] = len(comp)
    stats["n_literary"] = int(len(lit))
    stats["top10_cluster_sizes"] = [int(c) for _, c in top]
    stats["top10_cluster_words"] = [
        int(nwords[lit][cluster_of[lit] == cid].sum()) for cid, _ in top]
    # sample ids from the largest cluster for eyeballing
    big = top[0][0]
    sample = [all_rids[i] for i in lit if cluster_of[i] == big][:15]
    stats["largest_cluster_sample"] = sample

    t = pa.table({"rid": [all_rids[i] for i in lit],
                  "cluster": pa.array(cluster_of[lit], type=pa.int64()),
                  "nwords": pa.array(nwords[lit], type=pa.int32()),
                  "prefix": [all_pref[i] for i in lit]})
    pq.write_table(t, os.path.join(ROOT, "work", "clusters.parquet"),
                   compression="zstd")
    with open(os.path.join(ROOT, "work", "stage2_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
