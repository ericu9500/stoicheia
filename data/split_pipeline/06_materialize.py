#!/usr/bin/env python3
"""Stage 6: materialize all 10 folds to disk (single pass over the corpus).

Fold k:
  test  = bucket k (pristine) + PTEST (papyri via TM, inscriptions via PHI)
  val   = bucket (k+1)%10 + PVAL, cleaned against fold-k test zones
  train = everything else, with contaminated sentences excised and maximal
          runs of consecutive clean sentences re-stitched into segments

Test records are emitted verbatim. Val records are excised of any sentence
colliding with the fold's test zones (belt and braces for cluster misses).
Train records are excised of any sentence colliding with the fold's val+test
zones. Segments < MIN_SEG_CHARS (train: 100, val: 25) are dropped. Each
maximal run becomes its OWN record (no false word adjacencies across cuts).

Inscriptions: train-zone variants are emitted per-variant like normal text
records. PTEST/PVAL inscriptions are emitted as the full original row MINUS
the synthetic fields (synthetics of val/test PHI numbers appear nowhere).
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
from common import (fold_conflict_mask, N_BUCKETS, MASK_PTEST, MASK_PVAL,
                    ZONE_PTEST, ZONE_PVAL, ZONE_TRAIN, MIN_SEG_CHARS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))
MIN_VAL_CHARS = 25
INSCR_JSONL = os.path.join(ROOT, "raw", "Inscriptions_2",
                           "synthetic_editions_with_ithaca_text_fix.jsonl")
INSCR_REAL = ["edition", "with_diacritics", "without_diacritics", "ithaca_text"]
INSCR_SYNTH = {"synthetic", "synthetic_2"}


def test_mask(k):
    return (1 << k) | MASK_PTEST


def segments(text, starts, ends, masks, conflict, min_chars):
    """Yield (piece_text, was_cut) after excising sentences hitting conflict."""
    if not starts:
        return [], 0, 0
    keep = [not (m & conflict) for m in masks]
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


class FoldWriters:
    def __init__(self, base, tag):
        self.base = base
        self.tag = tag
        self.w = {}

    def get(self, fold, split):
        key = (fold, split)
        if key not in self.w:
            d = os.path.join(self.base, "fold_%d" % fold, "parts")
            os.makedirs(d, exist_ok=True)
            f = open(os.path.join(d, "%s-%s.jsonl.zst" % (split, self.tag)), "wb")
            cctx = zstd.ZstdCompressor(level=6)
            self.w[key] = (f, cctx.stream_writer(f))
        return self.w[key][1]

    def close(self):
        for f, s in self.w.values():
            s.close()
            f.close()


def emit(writer, obj):
    writer.write(orjson.dumps(obj) + b"\n")


def process_shard(args):
    tier, spath, mpath = args
    tag = tier + "-" + os.path.basename(spath).split(".")[0]
    st = pq.read_table(spath, columns=["rid", "source", "text", "starts", "ends"])
    mt = pq.read_table(mpath, columns=["rid", "zone", "masks"])
    assert st["rid"].to_pylist() == mt["rid"].to_pylist(), "shard misalignment"
    W = FoldWriters(OUTDIR, tag)
    stats = defaultdict(lambda: [0, 0, 0, 0])  # (fold,split,tier)->recs,chars,cut,dropped

    rids = st["rid"].to_pylist()
    sources = st["source"].to_pylist()
    texts = st["text"].to_pylist()
    starts_c = st["starts"].to_pylist()
    ends_c = st["ends"].to_pylist()
    zones = mt["zone"].to_pylist()
    masks_c = mt["masks"].to_pylist()

    for rid, source, text, starts, ends, zone, masks in zip(
            rids, sources, texts, starts_c, ends_c, zones, masks_c):
        inscr = tier == "inscriptions"
        for k in range(N_BUCKETS):
            if zone == k or zone == ZONE_PTEST:
                role = "test"
            elif zone == (k + 1) % N_BUCKETS or zone == ZONE_PVAL:
                role = "val"
            else:
                role = "train"
            if inscr and role != "train":
                continue  # val/test inscriptions are emitted centrally
            if role == "test":
                if tier != "pristine":
                    continue  # only pristine (incl. papyri) feeds test
                s = stats[(k, "test", tier)]
                emit(W.get(k, "test"), {"id": rid, "tier": tier,
                                        "source": source, "text": text})
                s[0] += 1
                s[1] += len(text)
                continue
            if role == "val" and tier != "pristine":
                continue
            conflict = test_mask(k) if role == "val" else fold_conflict_mask(k)
            min_chars = MIN_VAL_CHARS if role == "val" else MIN_SEG_CHARS
            pieces, cut, was_cut = segments(text, starts, ends, masks,
                                            conflict, min_chars)
            s = stats[(k, role, tier)]
            if not pieces:
                s[3] += 1
                s[2] += cut
                continue
            wr = W.get(k, role)
            if len(pieces) == 1 and not was_cut:
                emit(wr, {"id": rid, "tier": tier, "source": source,
                          "text": pieces[0]})
            else:
                for pi, piece in enumerate(pieces):
                    emit(wr, {"id": "%s#seg%d" % (rid, pi), "tier": tier,
                              "source": source, "text": piece})
            s[0] += len(pieces)
            s[1] += sum(len(p) for p in pieces)
            s[2] += cut
    W.close()
    return {"%d|%s|%s" % key: v for key, v in stats.items()}


def inscriptions_valtest():
    """Emit PTEST/PVAL inscriptions centrally: original row minus synthetics."""
    # per-PHI OR of all real-variant sentence masks (for val-vs-test cleaning)
    phi_mask = {}
    phi_zone = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "work", "masks",
                                           "inscriptions", "*.parquet"))):
        t = pq.read_table(p, columns=["rid", "zone", "masks"])
        for rid, zone, masks in zip(t["rid"].to_pylist(), t["zone"].to_pylist(),
                                    t["masks"].to_pylist()):
            phi, field = rid.split(":", 1)
            if zone not in (ZONE_PTEST, ZONE_PVAL) or field in INSCR_SYNTH:
                continue
            m = 0
            for x in masks:
                m |= x
            phi_mask[phi] = phi_mask.get(phi, 0) | m
            phi_zone[phi] = zone

    W = FoldWriters(OUTDIR, "inscr-central")
    stats = defaultdict(lambda: [0, 0, 0, 0])
    with open(INSCR_JSONL, "rb") as f:
        for line in f:
            try:
                r = orjson.loads(line)
            except orjson.JSONDecodeError:
                r = json.loads(line)  # pandas-written rows with bare NaN
            phi = "phi%s" % r["PHI_ID"]
            zone = phi_zone.get(phi)
            if zone is None:
                continue
            out = {"id": phi, "tier": "inscriptions", "source": "phi"}
            for fld, v in r.items():
                if fld not in INSCR_SYNTH:
                    out[fld] = v
            if zone == ZONE_PTEST:
                for k in range(N_BUCKETS):
                    emit(W.get(k, "test"), out)
                    s = stats[(k, "test", "inscriptions")]
                    s[0] += 1
            else:  # PVAL: drop from fold k's val if it collides with test zones
                m = phi_mask.get(phi, 0)
                for k in range(N_BUCKETS):
                    if m & test_mask(k):
                        stats[(k, "val", "inscriptions")][3] += 1
                        continue
                    emit(W.get(k, "val"), out)
                    stats[(k, "val", "inscriptions")][0] += 1
    W.close()
    return {"%d|%s|%s" % key: v for key, v in stats.items()}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    tasks = []
    for tier in ("pristine", "repaired", "bronze", "inscriptions"):
        for spath in sorted(glob.glob(os.path.join(
                ROOT, "work", "sentences", tier, "shard_*.parquet"))):
            mpath = os.path.join(ROOT, "work", "masks", tier,
                                 os.path.basename(spath))
            tasks.append((tier, spath, mpath))
    print("%d shards" % len(tasks), flush=True)

    agg = defaultdict(lambda: [0, 0, 0, 0])
    workers = max(4, os.cpu_count() - 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_shard, t) for t in tasks]
        futs.append(ex.submit(inscriptions_valtest))
        for i, f in enumerate(futs):
            for key, v in f.result().items():
                a = agg[key]
                for j in range(4):
                    a[j] += v[j]
            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(futs)), flush=True)

    with open(os.path.join(ROOT, "work", "stage6_stats.json"), "w") as f:
        json.dump(agg, f, indent=2)

    # concatenate parts (zstd frames concatenate losslessly)
    for k in range(N_BUCKETS):
        d = os.path.join(OUTDIR, "fold_%d" % k)
        pd = os.path.join(d, "parts")
        for split in ("train", "val", "test"):
            parts = sorted(glob.glob(os.path.join(pd, split + "-*.jsonl.zst")))
            outp = os.path.join(d, split + ".jsonl.zst")
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
        print("fold %d assembled" % k, flush=True)


if __name__ == "__main__":
    main()
