#!/usr/bin/env python3
"""Stage 7b: verify the documentary-clean corpus before training on it.

  1. census    stream train.jsonl.zst; FAIL if any record has source in
               {ddbdp, dclp, phi} or tier == inscriptions
  2. cleanness sample 1/SAMPLE_EVERY of emitted records; recompute their
               skeleton exact/bag/8-gram hashes and look them up in the
               stage-4b documentary index -> expect ZERO hits (proves the
               excision + re-stitching left no documentary trace)
  3. index     sanity: sample documentary records from work/sentences and
               confirm their keys DO hit the index (the net has no holes)

Exits non-zero on any failure.
"""
import glob
import json
import os
import random
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, skeleton, sentence_spans, NGRAM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "doc_clean")
OUTDIR = os.path.expandvars(os.environ.get("DOC_OUTDIR",
                        "$STOICHEIA_DATA"))
PAPYRI_SOURCES = {"ddbdp", "dclp"}
INSCR_SYNTH = {"synthetic", "synthetic_2"}
SAMPLE_EVERY = 25


def record_keys(text):
    """All contamination keys of one text: per-sentence exact+bag, stream 8-grams."""
    keys = []
    words_all = []
    for s, e in sentence_spans(text):
        sk = skeleton(text[s:e])
        w = sk.split()
        if not w:
            continue
        keys.append(h64(sk))
        keys.append(h64(" ".join(sorted(w))))
        words_all.extend(w)
    for i in range(len(words_all) - NGRAM + 1):
        keys.append(h64(" ".join(words_all[i:i + NGRAM])))
    return keys


def main():
    import orjson
    fails = []
    keys_idx = np.load(os.path.join(OUT, "index_keys.npy"), mmap_mode="r")

    def hits(qkeys):
        if not qkeys:
            return 0
        q = np.array(qkeys, dtype=np.uint64)
        pos = np.searchsorted(keys_idx, q)
        pos[pos >= len(keys_idx)] = len(keys_idx) - 1
        return int((keys_idx[pos] == q).sum())

    # ---- 1 + 2: census and cleanness over the emitted corpus ----
    proc = subprocess.Popen(["zstdcat", os.path.join(OUTDIR, "train.jsonl.zst")],
                            stdout=subprocess.PIPE)
    n = 0
    census = {}
    bad_source = 0
    sampled = checked_keys = hit_keys = 0
    hit_examples = []
    for line in proc.stdout:
        r = orjson.loads(line)
        n += 1
        census[(r["tier"], r["source"])] = census.get((r["tier"], r["source"]), 0) + 1
        if r["source"] in PAPYRI_SOURCES or r["source"] == "phi" \
                or r["tier"] == "inscriptions":
            bad_source += 1
        if n % SAMPLE_EVERY == 0:
            ks = record_keys(r["text"])
            h = hits(ks)
            sampled += 1
            checked_keys += len(ks)
            hit_keys += h
            if h and len(hit_examples) < 5:
                hit_examples.append(r["id"])
    if proc.wait() != 0:
        raise RuntimeError("zstdcat failed")
    if bad_source:
        fails.append("census: %d documentary records in output" % bad_source)
    if hit_keys:
        fails.append("cleanness: %d/%d sampled keys hit the doc index (e.g. %s)"
                     % (hit_keys, checked_keys, hit_examples))
    print("records: %d  sampled: %d  keys checked: %d  doc-index hits: %d"
          % (n, sampled, checked_keys, hit_keys))
    by_tier = {}
    for (t, s), c in census.items():
        by_tier[t] = by_tier.get(t, 0) + c
    print("by tier:", json.dumps(by_tier))

    # ---- 3: the index actually contains documentary text ----
    rng = random.Random(0)
    pos_checked = pos_hit = 0
    for tier, want_src in (("inscriptions", None), ("pristine", PAPYRI_SOURCES)):
        shards = sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet")))
        for p in rng.sample(shards, min(3, len(shards))):
            t = pq.read_table(p, columns=["rid", "source", "text"])
            rows = [(r, s, x) for r, s, x in zip(t["rid"].to_pylist(),
                                                 t["source"].to_pylist(),
                                                 t["text"].to_pylist())
                    if (want_src is None and (":" not in r or
                        r.split(":", 1)[1] not in INSCR_SYNTH))
                    or (want_src is not None and s in want_src)]
            for r, s, x in rng.sample(rows, min(50, len(rows))):
                ks = record_keys(x)
                if not ks:
                    continue
                pos_checked += 1
                if hits(ks):
                    pos_hit += 1
    print("index sanity: %d/%d documentary samples hit the index"
          % (pos_hit, pos_checked))
    if pos_checked and pos_hit < pos_checked * 0.98:
        fails.append("index sanity: only %d/%d documentary samples found in index"
                     % (pos_hit, pos_checked))

    if fails:
        print("\nVERIFY FAILED:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
