#!/usr/bin/env python3
"""Stage 1: normalize + sentence-tokenize every record of every tier.

Outputs work/sentences/<tier>/shard_NNNN.parquet with columns:
  rid    : unique record id (inscriptions: "phi<ID>:<field>")
  source : provenance tag
  zone   : int8 preliminary zone (-1 = literary pristine, assigned in stage 3;
           0-9 unused here; 10=PTEST, 11=PVAL, 12=TRAIN)
  text   : original text of the unit
  starts, ends : list<int32> sentence spans (concatenation of spans == text)
  skels  : list<str> per-sentence normalized skeletons

Zone rule (fixed across folds, Ithaca-compatible): PHI/TM number ending in
3 -> PTEST, 4 -> PVAL, anything else -> TRAIN. Applied to source ddbdp/dclp in
ANY tier and to all Inscriptions_2 records. ddbdp ids carry no TM; they are
joined to TM via the original extraction files (99.97% coverage);
the remainder falls back to xxhash64(id) last digit (counted in stats).
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import skeleton, sentence_spans, ZONE_PTEST, ZONE_PVAL, ZONE_TRAIN
import xxhash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "work", "sentences")

DDBDP_JSONL = os.path.expandvars("$STOICHEIA_DATA/clean/ddbdp.jsonl")
PAPYRI_TM_JSONL = os.path.expandvars("$STOICHEIA_DATA/data/papyri_clean.jsonl")
INSCR_JSONL = os.path.join(RAW, "Inscriptions_2", "synthetic_editions_with_ithaca_text_fix.jsonl")
INSCR_FIELDS = ["edition", "with_diacritics", "without_diacritics",
                "synthetic", "synthetic_2", "ithaca_text"]

SCHEMA = pa.schema([
    ("rid", pa.string()), ("source", pa.string()), ("zone", pa.int8()),
    ("text", pa.string()),
    ("starts", pa.list_(pa.int32())), ("ends", pa.list_(pa.int32())),
    ("skels", pa.list_(pa.string())),
])

ROWS_PER_SHARD = 25000


def build_tm_map():
    tm_by_base = {}
    with open(PAPYRI_TM_JSONL, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            tm_by_base[r["file"]] = str(r["TM"])
    id2tm = {}
    with open(DDBDP_JSONL, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            base = r["file"].rsplit("/", 1)[-1]
            tm = tm_by_base.get(base)
            if tm is not None:
                id2tm[r["id"]] = tm
    return id2tm


def digit_zone(numstr):
    d = numstr.rstrip()[-1]
    if d == "3":
        return ZONE_PTEST
    if d == "4":
        return ZONE_PVAL
    return ZONE_TRAIN


def zone_for(source, rid, tier, id2tm, stats):
    if source == "dclp":
        return digit_zone(rid.split("_")[0])
    if source == "ddbdp":
        tm = id2tm.get(rid) or id2tm.get(rid.split("#")[0])
        if tm is None:
            stats["ddbdp_tm_fallback"] = stats.get("ddbdp_tm_fallback", 0) + 1
            return digit_zone(str(xxhash.xxh64_intdigest(rid) % 10))
        return digit_zone(tm)
    if source == "phi":
        return digit_zone(rid.split(":")[0].replace("phi", ""))
    if tier == "pristine":
        return -1  # literary pristine: bucket assigned in stage 3
    return ZONE_TRAIN


def tokenize_row(rid, source, zone, text, cols):
    spans = sentence_spans(text)
    starts, ends, skels = [], [], []
    for s, e in spans:
        sk = skeleton(text[s:e])
        if not sk:
            # keep coverage: merge into previous span
            if ends:
                ends[-1] = e
            continue
        starts.append(s)
        ends.append(e)
        skels.append(sk)
    cols["rid"].append(rid)
    cols["source"].append(source)
    cols["zone"].append(zone)
    cols["text"].append(text)
    cols["starts"].append(starts)
    cols["ends"].append(ends)
    cols["skels"].append(skels)


def new_cols():
    return {k: [] for k in ("rid", "source", "zone", "text", "starts", "ends", "skels")}


def flush(cols, tier, shard_idx):
    if not cols["rid"]:
        return 0
    t = pa.table(cols, schema=SCHEMA)
    os.makedirs(os.path.join(OUT, tier), exist_ok=True)
    pq.write_table(t, os.path.join(OUT, tier, "shard_%05d.parquet" % shard_idx),
                   compression="zstd")
    return len(cols["rid"])


# ------------------------------------------------------------ task workers
def do_parquet_task(args):
    tier, path, rg, offset, length, shard_idx, id2tm = args
    stats = {"records": 0, "sentences": 0, "zones": {}}
    pf = pq.ParquetFile(path)
    t = pf.read_row_group(rg, columns=["source", "id", "text"])
    t = t.slice(offset, length)
    cols = new_cols()
    for source, rid, text in zip(t["source"].to_pylist(), t["id"].to_pylist(),
                                 t["text"].to_pylist()):
        z = zone_for(source, rid, tier, id2tm, stats)
        tokenize_row(rid, source, z, text, cols)
        stats["records"] += 1
        stats["sentences"] += len(cols["skels"][-1])
        stats["zones"][z] = stats["zones"].get(z, 0) + 1
    n_written = 0
    # split into multiple shards if the row group is large
    n = len(cols["rid"])
    for off in range(0, n, ROWS_PER_SHARD):
        part = {k: v[off:off + ROWS_PER_SHARD] for k, v in cols.items()}
        flush(part, tier, shard_idx + off // ROWS_PER_SHARD)
        n_written += 1
    return stats


def loads_lenient(line):
    """orjson rejects bare NaN (pandas-written rows); stdlib json accepts it."""
    try:
        return orjson.loads(line)
    except orjson.JSONDecodeError:
        return json.loads(line)


def do_jsonl_task(args):
    tier, path, byte_start, byte_end, shard_idx = args
    stats = {"records": 0, "sentences": 0, "zones": {}}
    cols = new_cols()
    shard_off = 0
    with open(path, "rb") as f:
        f.seek(byte_start)
        if byte_start > 0:
            f.readline()  # skip partial line (owned by previous chunk)
        while f.tell() <= byte_end:
            line = f.readline()
            if not line:
                break
            r = loads_lenient(line)
            if tier == "bronze":
                units = [(r["id"], "bronze", r["text"])]
            else:  # inscriptions
                phi = str(r["PHI_ID"])
                units = [("phi%s:%s" % (phi, fld), "phi", r[fld])
                         for fld in INSCR_FIELDS
                         if isinstance(r.get(fld), str) and r[fld].strip()]
            for rid, source, text in units:
                z = zone_for(source, rid, tier, None, stats)
                tokenize_row(rid, source, z, str(text), cols)
                stats["records"] += 1
                stats["sentences"] += len(cols["skels"][-1])
                stats["zones"][z] = stats["zones"].get(z, 0) + 1
            if len(cols["rid"]) >= ROWS_PER_SHARD:
                flush(cols, tier, shard_idx + shard_off)
                shard_off += 1
                cols = new_cols()
    flush(cols, tier, shard_idx + shard_off)
    return stats


def jsonl_tasks(tier, path, n_chunks, shard_base, shards_per_chunk=400):
    size = os.path.getsize(path)
    step = size // n_chunks + 1
    tasks = []
    for i in range(n_chunks):
        tasks.append((tier, path, i * step, min((i + 1) * step, size) - 1,
                      shard_base + i * shards_per_chunk))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(4, os.cpu_count() - 8))
    ap.add_argument("--tiers", default="pristine,repaired,bronze,inscriptions")
    args = ap.parse_args()
    tiers = args.tiers.split(",")

    id2tm = build_tm_map()
    print("ddbdp id->TM map: %d entries" % len(id2tm), flush=True)

    tasks = []
    if "pristine" in tiers or "repaired" in tiers:
        for tier in ("pristine", "repaired"):
            if tier not in tiers:
                continue
            shard_idx = 0
            for path in sorted(glob.glob(os.path.join(
                    RAW, "AncientGreek", "data", tier, "*.parquet"))):
                pf = pq.ParquetFile(path)
                for rg in range(pf.num_row_groups):
                    nrows = pf.metadata.row_group(rg).num_rows
                    for off in range(0, nrows, ROWS_PER_SHARD):
                        ln = min(ROWS_PER_SHARD, nrows - off)
                        tasks.append((tier, "pq",
                                      (tier, path, rg, off, ln, shard_idx, id2tm)))
                        shard_idx += 1
    if "bronze" in tiers:
        tasks += [("bronze", "jl", t) for t in
                  jsonl_tasks("bronze", os.path.join(RAW, "bronze.jsonl"), 96, 0)]
    if "inscriptions" in tiers:
        tasks += [("inscriptions", "jl", t) for t in
                  jsonl_tasks("inscriptions", INSCR_JSONL, 48, 0)]

    print("%d tasks" % len(tasks), flush=True)
    agg = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for tier, kind, t in tasks:
            fn = do_parquet_task if kind == "pq" else do_jsonl_task
            futs.append((tier, ex.submit(fn, t)))
        for i, (tier, f) in enumerate(futs):
            st = f.result()
            a = agg.setdefault(tier, {"records": 0, "sentences": 0, "zones": {},
                                      "ddbdp_tm_fallback": 0})
            a["records"] += st["records"]
            a["sentences"] += st["sentences"]
            a["ddbdp_tm_fallback"] += st.get("ddbdp_tm_fallback", 0)
            for z, c in st["zones"].items():
                a["zones"][z] = a["zones"].get(z, 0) + c
            if (i + 1) % 25 == 0:
                print("  %d/%d tasks done" % (i + 1, len(futs)), flush=True)

    os.makedirs(os.path.join(ROOT, "work"), exist_ok=True)
    with open(os.path.join(ROOT, "work", "stage1_stats.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
