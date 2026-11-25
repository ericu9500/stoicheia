#!/usr/bin/env python3
"""Stage 6b (documentary-clean variant): materialize ONE train corpus that is
clean of all documentary text and every textual trace of it.

Output record set = pristine + repaired + bronze, where:
  - records with source ddbdp/dclp (papyri, both tiers) are DROPPED entirely
  - (inscriptions tier is simply never read -- it is excluded by construction)
  - every sentence whose stage-5b mask is nonzero (i.e. matches ANY PHI
    inscription or papyrus by exact skeleton / bag / shared word-8-gram) is
    EXCISED; maximal clean runs are re-stitched into id#segN segments of
    >= MIN_SEG_CHARS chars (same rule as the 10-fold train sets)
  - NO literary-bucket excision: all 10 literary buckets are trainable here
    (this corpus is fold-free; it is held out only against documentary text)

Output: <DOC_OUTDIR>/train.jsonl.zst  (records {id, tier, source, text})
Default DOC_OUTDIR: $STOICHEIA_DATA
"""
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import orjson
import pyarrow.parquet as pq
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MIN_SEG_CHARS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "doc_clean")
OUTDIR = os.path.expandvars(os.environ.get("DOC_OUTDIR",
                        "$STOICHEIA_DATA"))
PAPYRI_SOURCES = {"ddbdp", "dclp"}


def segments(text, starts, ends, masks, min_chars):
    """(pieces, cut_chars, was_cut) after excising sentences with nonzero mask."""
    if not starts:
        return [], 0, 0
    keep = [not m for m in masks]
    if all(keep):
        return [text], 0, 0
    pieces = []
    cut_chars = 0
    i = 0
    n = len(starts)
    while i < n:
        if not keep[i]:
            cut_chars += ends[i] - starts[i]
            i += 1
            continue
        j = i
        while j + 1 < n and keep[j + 1]:
            j += 1
        piece = text[starts[i]:ends[j]].strip()
        if len(piece) >= min_chars:
            pieces.append(piece)
        else:
            cut_chars += len(piece)
        i = j + 1
    return pieces, cut_chars, 1


def process_shard(args):
    tier, spath, mpath = args
    tag = tier + "-" + os.path.basename(spath).split(".")[0]
    st = pq.read_table(spath, columns=["rid", "source", "text", "starts", "ends"])
    mt = pq.read_table(mpath, columns=["rid", "masks"])
    assert st["rid"].to_pylist() == mt["rid"].to_pylist(), "shard misalignment"

    pd = os.path.join(OUTDIR, "parts")
    os.makedirs(pd, exist_ok=True)
    f = open(os.path.join(pd, "train-%s.jsonl.zst" % tag), "wb")
    w = zstd.ZstdCompressor(level=6).stream_writer(f)
    stats = defaultdict(lambda: [0, 0, 0, 0])  # tier -> recs,chars,cut,dropped

    for rid, source, text, starts, ends, masks in zip(
            st["rid"].to_pylist(), st["source"].to_pylist(),
            st["text"].to_pylist(), st["starts"].to_pylist(),
            st["ends"].to_pylist(), mt["masks"].to_pylist()):
        s = stats[tier]
        if source in PAPYRI_SOURCES:
            s[3] += 1
            continue
        pieces, cut, was_cut = segments(text, starts, ends, masks, MIN_SEG_CHARS)
        if not pieces:
            s[3] += 1
            s[2] += cut
            continue
        if len(pieces) == 1 and not was_cut:
            w.write(orjson.dumps({"id": rid, "tier": tier, "source": source,
                                  "text": pieces[0]}) + b"\n")
        else:
            for pi, piece in enumerate(pieces):
                w.write(orjson.dumps({"id": "%s#seg%d" % (rid, pi), "tier": tier,
                                      "source": source, "text": piece}) + b"\n")
        s[0] += len(pieces)
        s[1] += sum(len(p) for p in pieces)
        s[2] += cut
    w.close()
    f.close()
    return dict(stats)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = []
    for tier in ("pristine", "repaired", "bronze"):
        for spath in sorted(glob.glob(os.path.join(
                ROOT, "work", "sentences", tier, "shard_*.parquet"))):
            mpath = os.path.join(OUT, "masks", tier, os.path.basename(spath))
            tasks.append((tier, spath, mpath))
    print("%d shards" % len(tasks), flush=True)

    agg = defaultdict(lambda: [0, 0, 0, 0])
    workers = max(4, min(16, (os.cpu_count() or 12) - 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(process_shard, tasks, chunksize=1):
            for key, v in res.items():
                a = agg[key]
                for j in range(4):
                    a[j] += v[j]

    with open(os.path.join(OUT, "stage6b_stats.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2), flush=True)

    # concatenate parts (zstd frames concatenate losslessly)
    pd = os.path.join(OUTDIR, "parts")
    parts = sorted(glob.glob(os.path.join(pd, "train-*.jsonl.zst")))
    outp = os.path.join(OUTDIR, "train.jsonl.zst")
    with open(outp, "wb") as out:
        for p in parts:
            with open(p, "rb") as src:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    out.write(chunk)
    for p in parts:
        os.remove(p)
    os.rmdir(pd)
    print("assembled %s" % outp, flush=True)


if __name__ == "__main__":
    main()
