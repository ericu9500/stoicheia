#!/usr/bin/env python3
"""Stage 12: rare-word paraphrase screen on the remaining bronze (FILTER).

A translation/paraphrase of a specific Greek text shares its RARE vocabulary
(proper names, unusual terms) even when no 8-gram survives verbatim matching.
For EVERY fold: build the rare-word inventory of the fold's test records
(document frequency <= DF_MAX, length >= 6), then DROP from train every
bronze record that shares >= MIN_SHARED rare words with any single test
record. Over-exclusion (topical coincidence) is accepted by design.

Runs after stage 11; verify (stage 7) re-certifies afterwards.
Output: work/stage12_stats.json (+ samples of what was dropped).
"""
import io
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import orjson
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import skeleton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))
DF_MAX = 3          # word is 'rare' if in <= DF_MAX test records
MIN_SHARED = 4      # shared rare words with ONE test record => drop


def read_zst_lines(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    yield line


def filter_fold(k):
    d = os.path.join(OUTDIR, "fold_%d" % k)
    df = Counter()
    test_words = []
    for line in read_zst_lines(os.path.join(d, "test.jsonl.zst")):
        rec = orjson.loads(line)
        text = rec.get("text") or rec.get("with_diacritics") or ""
        ws = set(skeleton(text).split())
        test_words.append((rec["id"], ws))
        for w in ws:
            df[w] += 1
    rare = {w for w, c in df.items() if c <= DF_MAX and len(w) >= 6}
    inv = defaultdict(list)
    for i, (rid, ws) in enumerate(test_words):
        for w in ws & rare:
            inv[w].append(i)

    src = os.path.join(d, "train.jsonl.zst")
    tmp = src + ".tmp"
    cctx = zstd.ZstdCompressor(level=6)
    kept = dropped = 0
    dropped_chars = 0
    samples = []
    with open(tmp, "wb") as fout:
        writer = cctx.stream_writer(fout)
        for line in read_zst_lines(src):
            rec = orjson.loads(line)
            if rec.get("tier") == "bronze":
                ws = set(skeleton(rec.get("text", "")).split()) & rare
                if len(ws) >= MIN_SHARED:
                    per_test = Counter()
                    for w in ws:
                        for i in inv[w]:
                            per_test[i] += 1
                    best_i, best_c = per_test.most_common(1)[0]
                    if best_c >= MIN_SHARED:
                        dropped += 1
                        dropped_chars += len(rec.get("text", ""))
                        if len(samples) < 5:
                            samples.append({"bronze_id": rec["id"],
                                            "test_id": test_words[best_i][0],
                                            "shared": best_c})
                        continue
            kept += 1
            writer.write(line.encode())
        writer.close()
    os.replace(tmp, src)
    return k, {"kept": kept, "dropped_bronze": dropped,
               "dropped_Mchars": round(dropped_chars / 1e6, 2),
               "rare_words": len(rare), "samples": samples}


def main():
    stats = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for k, res in ex.map(filter_fold, range(10)):
            stats["fold_%d" % k] = res
            print("fold %d: dropped %d bronze recs (%.2f Mchars), kept %d"
                  % (k, res["dropped_bronze"], res["dropped_Mchars"],
                     res["kept"]), flush=True)
    with open(os.path.join(ROOT, "work", "stage12_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
