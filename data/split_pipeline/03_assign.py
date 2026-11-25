#!/usr/bin/env python3
"""Stage 3: assign literary pristine clusters to the 10 rotating buckets.

Greedy bin-packing by cluster word count (largest first, ties shuffled with a
fixed seed) into the currently lightest bucket -> each bucket ~10% of literary
pristine words. Whole clusters move together, so all editions/duplicates of a
text share one bucket.

Guard: a cluster larger than SPLIT_FRAC of a bucket is split into its
id-prefix groups (volumes/works) which are then packed independently. Any
resulting cross-bucket duplication is excised from train by stage 5/6 masks,
so this trades a little data for balanced folds without leaking.

Output: work/literary_zones.parquet (rid, zone, cluster) + stage3_stats.json
"""
import heapq
import json
import os
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260709
N_BUCKETS = 10
SPLIT_FRAC = float(os.environ.get("SPLIT_FRAC", "0.5"))  # of one bucket

t = pq.read_table(os.path.join(ROOT, "work", "clusters.parquet"))
rids = t["rid"].to_pylist()
clusters = t["cluster"].to_numpy()
nwords = t["nwords"].to_numpy()
prefixes = t["prefix"].to_pylist()

total_words = int(nwords.sum())
cap = total_words / N_BUCKETS * SPLIT_FRAC

# build assignable units: whole clusters, or prefix groups of huge clusters
cw = defaultdict(int)
for c, w in zip(clusters, nwords):
    cw[int(c)] += int(w)

unit_of_row = [None] * len(rids)     # unit key per record
unit_words = defaultdict(int)
n_split = 0
for i, (c, w, pf) in enumerate(zip(clusters, nwords, prefixes)):
    c = int(c)
    if cw[c] > cap:
        key = ("split", c, pf)
    else:
        key = ("whole", c, "")
    unit_of_row[i] = key
    unit_words[key] += int(w)
n_split = len({k for k in unit_words if k[0] == "split"})
split_clusters = len({k[1] for k in unit_words if k[0] == "split"})

rng = np.random.RandomState(SEED)
order = sorted(unit_words.keys(), key=lambda k: (-unit_words[k], rng.rand()))
heap = [(0, b) for b in range(N_BUCKETS)]
heapq.heapify(heap)
bucket_of = {}
for u in order:
    w, b = heapq.heappop(heap)
    bucket_of[u] = b
    heapq.heappush(heap, (w + unit_words[u], b))

zones = np.array([bucket_of[u] for u in unit_of_row], dtype=np.int8)
out = pa.table({"rid": rids, "zone": pa.array(zones, type=pa.int8()),
                "cluster": pa.array(clusters, type=pa.int64())})
pq.write_table(out, os.path.join(ROOT, "work", "literary_zones.parquet"),
               compression="zstd")

bucket_words = defaultdict(int)
bucket_recs = defaultdict(int)
for z, w in zip(zones, nwords):
    bucket_words[int(z)] += int(w)
    bucket_recs[int(z)] += 1
stats = {"bucket_words": dict(sorted(bucket_words.items())),
         "bucket_records": dict(sorted(bucket_recs.items())),
         "n_clusters": len(cw),
         "clusters_split": split_clusters,
         "split_units": n_split,
         "largest_cluster_words": max(cw.values()),
         "total_words": total_words}
with open(os.path.join(ROOT, "work", "stage3_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
