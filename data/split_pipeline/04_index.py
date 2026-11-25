#!/usr/bin/env python3
"""Stage 4: global contamination index.

For every record that can ever be val/test (all pristine records in buckets
0-9 or PTEST/PVAL, plus Inscriptions_2 variants in PTEST/PVAL) emit hash keys:
  - exact skeleton hash of each sentence
  - bag hash (sorted words) of each sentence          [catches reorderings]
  - every word-8-gram hash over the record's whole word stream
    (cross-sentence, so differing punctuation/splits can't hide a quote)
each mapped to the record's zone bit. Keys are merged by bitwise OR.

Output: work/index_keys.npy (sorted uint64), work/index_masks.npy (uint16)
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, NGRAM, ZONE_TRAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "work", "index_parts")


def load_literary_zones():
    t = pq.read_table(os.path.join(ROOT, "work", "literary_zones.parquet"),
                      columns=["rid", "zone"])
    return dict(zip(t["rid"].to_pylist(), t["zone"].to_pylist()))


def process_shard(args):
    tier, path, out_path = args
    zf = ZONES_CACHE.get("z")
    t = pq.read_table(path, columns=["rid", "zone", "skels"])
    keys, masks = [], []
    n_contrib = 0
    for rid, _z1, skels in zip(t["rid"].to_pylist(), t["zone"].to_pylist(),
                               t["skels"].to_pylist()):
        zone = zf.get(rid, -1)
        if zone < 0 or zone >= ZONE_TRAIN:
            continue  # train-always records never appear in val/test
        bit = 1 << zone
        n_contrib += 1
        words_all = []
        for sk in skels:
            w = sk.split()
            keys.append(h64(sk))
            masks.append(bit)
            keys.append(h64(" ".join(sorted(w))))
            masks.append(bit)
            words_all.extend(w)
        if len(words_all) >= NGRAM:
            for i in range(len(words_all) - NGRAM + 1):
                keys.append(h64(" ".join(words_all[i:i + NGRAM])))
                masks.append(bit)
    k = np.array(keys, dtype=np.uint64)
    m = np.array(masks, dtype=np.uint16)
    np.savez(out_path, k=k, m=m)
    return len(k), n_contrib


ZONES_CACHE = {}


def init_worker(zpath):
    import pyarrow.parquet as pq2
    t = pq2.read_table(zpath, columns=["rid", "zone"])
    ZONES_CACHE["z"] = dict(zip(t["rid"].to_pylist(), t["zone"].to_pylist()))


def main():
    os.makedirs(PARTS, exist_ok=True)
    zpath = os.path.join(ROOT, "work", "zones_final.parquet")
    tasks = []
    for tier in ("pristine", "inscriptions"):
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            out = os.path.join(PARTS, "%s_%s.npz" %
                               (tier, os.path.basename(p).split(".")[0]))
            tasks.append((tier, p, out))
    print("%d shards" % len(tasks), flush=True)

    workers = max(4, os.cpu_count() - 8)
    total_keys = 0
    total_contrib = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(zpath,)) as ex:
        for nk, nc in ex.map(process_shard, tasks, chunksize=1):
            total_keys += nk
            total_contrib += nc
    print("raw keys: %d from %d contributing records" %
          (total_keys, total_contrib), flush=True)

    # merge: concat -> sort -> OR-reduce duplicate keys
    parts = sorted(glob.glob(os.path.join(PARTS, "*.npz")))
    ks, ms = [], []
    for p in parts:
        z = np.load(p)
        ks.append(z["k"])
        ms.append(z["m"])
    k = np.concatenate(ks)
    m = np.concatenate(ms)
    del ks, ms
    order = np.argsort(k, kind="stable")
    k = k[order]
    m = m[order]
    del order
    boundary = np.empty(len(k), dtype=bool)
    boundary[0] = True
    np.not_equal(k[1:], k[:-1], out=boundary[1:])
    starts = np.flatnonzero(boundary)
    uk = k[starts]
    um = np.bitwise_or.reduceat(m, starts)
    np.save(os.path.join(ROOT, "work", "index_keys.npy"), uk)
    np.save(os.path.join(ROOT, "work", "index_masks.npy"), um)
    stats = {"raw_keys": int(total_keys), "unique_keys": int(len(uk)),
             "contributing_records": int(total_contrib)}
    with open(os.path.join(ROOT, "work", "stage4_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
