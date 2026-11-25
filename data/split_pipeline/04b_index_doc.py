#!/usr/bin/env python3
"""Stage 4b (documentary-clean variant): documentary-only contamination index.

Unlike stage 4 (which indexes only zone 0-11 material, i.e. digit-3/4
documentary), this indexes EVERY documentary record regardless of PHI/TM digit:
  - all PHI inscription real-edition variants (tier inscriptions; the
    synthetic/synthetic_2 fields are skipped -- they are model-generated
    derivatives of the edition, not attested text)
  - all papyri (source ddbdp/dclp) in BOTH the pristine and repaired tiers

Keys per record (same recipe as stage 4): exact sentence-skeleton hash, bag
(sorted-words) hash, every word-8-gram hash over the whole word stream. Every
key maps to MASK_DOC (bit 13; bits 0-11 are the fold zones, so a separate bit
keeps this index unambiguous).

Output: work/doc_clean/index_keys.npy (sorted uint64),
        work/doc_clean/index_masks.npy (uint16)
Nothing under the original work/ artifacts is touched.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, NGRAM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "doc_clean")
PARTS = os.path.join(OUT, "index_parts")
MASK_DOC = np.uint16(1 << 13)
PAPYRI_SOURCES = {"ddbdp", "dclp"}
INSCR_SYNTH = {"synthetic", "synthetic_2"}


def process_shard(args):
    tier, path, out_path = args
    t = pq.read_table(path, columns=["rid", "source", "skels"])
    keys = []
    n_contrib = 0
    for rid, source, skels in zip(t["rid"].to_pylist(), t["source"].to_pylist(),
                                  t["skels"].to_pylist()):
        if tier == "inscriptions":
            field = rid.split(":", 1)[1] if ":" in rid else ""
            if field in INSCR_SYNTH:
                continue
        elif source not in PAPYRI_SOURCES:
            continue
        n_contrib += 1
        words_all = []
        for sk in skels:
            w = sk.split()
            keys.append(h64(sk))
            keys.append(h64(" ".join(sorted(w))))
            words_all.extend(w)
        if len(words_all) >= NGRAM:
            for i in range(len(words_all) - NGRAM + 1):
                keys.append(h64(" ".join(words_all[i:i + NGRAM])))
    np.save(out_path, np.array(keys, dtype=np.uint64))
    return len(keys), n_contrib


def main():
    os.makedirs(PARTS, exist_ok=True)
    tasks = []
    for tier in ("pristine", "repaired", "inscriptions"):
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            out = os.path.join(PARTS, "%s_%s.npy" %
                               (tier, os.path.basename(p).split(".")[0]))
            tasks.append((tier, p, out))
    print("%d shards" % len(tasks), flush=True)

    workers = max(4, min(16, (os.cpu_count() or 12) - 8))
    total_keys = 0
    total_contrib = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nk, nc in ex.map(process_shard, tasks, chunksize=1):
            total_keys += nk
            total_contrib += nc
    print("raw keys: %d from %d documentary records" %
          (total_keys, total_contrib), flush=True)

    ks = [np.load(p) for p in sorted(glob.glob(os.path.join(PARTS, "*.npy")))]
    k = np.unique(np.concatenate(ks))
    del ks
    np.save(os.path.join(OUT, "index_keys.npy"), k)
    np.save(os.path.join(OUT, "index_masks.npy"),
            np.full(len(k), MASK_DOC, dtype=np.uint16))
    stats = {"raw_keys": int(total_keys), "unique_keys": int(len(k)),
             "contributing_records": int(total_contrib)}
    with open(os.path.join(OUT, "stage4b_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
