#!/usr/bin/env python3
"""Stage 3b: canonical zone map for EVERY record of every tier.

Single source of truth consumed by stages 4/5/6 (shard-level `zone` columns
are only preliminary). Rules:
  - literary pristine: bucket from stage 3
  - papyri (ddbdp/dclp, ANY tier): TM digit rule, with chunk suffixes (#N)
    stripped before the TM lookup  [doublecheck D2 fix]
  - inscriptions: PHI digit rule (from stage 1 zones)
  - repaired literary: the bucket of its pristine sibling work/volume (same
    work_prefix) when one exists, else TRAIN  [doublecheck D3 fix: repaired
    pages of a volume leave train exactly when the volume is val/test]
  - bronze: TRAIN

Output: work/zones_final.parquet (rid, zone int8) over all tiers.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import xxhash
import orjson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ZONE_PTEST, ZONE_PVAL, ZONE_TRAIN, work_prefix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDBDP_JSONL = os.path.expandvars("$STOICHEIA_DATA/clean/ddbdp.jsonl")
PAPYRI_TM_JSONL = os.path.expandvars("$STOICHEIA_DATA/data/papyri_clean.jsonl")


def digit_zone(numstr):
    d = str(numstr).rstrip()[-1]
    if d == "3":
        return ZONE_PTEST
    if d == "4":
        return ZONE_PVAL
    return ZONE_TRAIN


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
            tm = tm_by_base.get(r["file"].rsplit("/", 1)[-1])
            if tm is not None:
                id2tm[r["id"]] = tm
    return id2tm


def main():
    id2tm = build_tm_map()

    # literary pristine buckets + prefix -> bucket map
    lz = pq.read_table(os.path.join(ROOT, "work", "literary_zones.parquet"),
                       columns=["rid", "zone"])
    lit_zone = dict(zip(lz["rid"].to_pylist(),
                        [int(z) for z in lz["zone"].to_pylist()]))
    cl = pq.read_table(os.path.join(ROOT, "work", "clusters.parquet"),
                       columns=["rid", "prefix"])
    prefix_bucket = {}
    conflicts = 0
    for rid, pf in zip(cl["rid"].to_pylist(), cl["prefix"].to_pylist()):
        b = lit_zone.get(rid)
        if b is None:
            continue
        if pf in prefix_bucket and prefix_bucket[pf] != b:
            conflicts += 1
        prefix_bucket[pf] = b

    stats = Counter()
    rids_all, zones_all = [], []
    seen = set()
    dup_rids = 0
    for tier in ("pristine", "repaired", "bronze", "inscriptions"):
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            t = pq.read_table(p, columns=["rid", "source", "zone"])
            for rid, src, z1 in zip(t["rid"].to_pylist(),
                                    t["source"].to_pylist(),
                                    t["zone"].to_pylist()):
                base = rid.split("#")[0]
                if src == "dclp":
                    z = digit_zone(base.split("_")[0])
                elif src == "ddbdp":
                    tm = id2tm.get(base) or id2tm.get(base.split("_")[0])
                    if tm is None:
                        stats["ddbdp_tm_fallback_" + tier] += 1
                        z = digit_zone(str(xxhash.xxh64_intdigest(base) % 10))
                    else:
                        z = digit_zone(tm)
                elif src == "phi":
                    z = int(z1)                      # PHI digit from stage 1
                elif tier == "pristine":
                    z = lit_zone.get(rid, ZONE_TRAIN)
                    if rid not in lit_zone:
                        stats["literary_unassigned"] += 1
                elif tier == "repaired":
                    z = prefix_bucket.get(work_prefix(src, rid), ZONE_TRAIN)
                    if z != ZONE_TRAIN:
                        stats["repaired_sibling_bucketed"] += 1
                else:
                    z = ZONE_TRAIN
                if rid in seen:
                    dup_rids += 1
                seen.add(rid)
                rids_all.append(rid)
                zones_all.append(z)
                stats["zone_%d_%s" % (z, tier)] += 1

    out = pa.table({"rid": rids_all,
                    "zone": pa.array(zones_all, type=pa.int8())})
    pq.write_table(out, os.path.join(ROOT, "work", "zones_final.parquet"),
                   compression="zstd")
    summary = {"records": len(rids_all), "duplicate_rids_across_tiers": dup_rids,
               "prefix_bucket_conflicts": conflicts,
               "counts": dict(sorted(stats.items()))}
    with open(os.path.join(ROOT, "work", "stage3b_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
