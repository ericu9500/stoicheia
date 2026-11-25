"""Verify one fold's shards before training on them.

Checks, per fold:
  1. census      shard index row counts / letter sums per tier vs a fresh streaming pass
                 over the source train.jsonl.zst (expect small drops from the <95%-Greek
                 normalize filter, nothing else)
  2. leakage     no shard record id (base id, #segN stripped) appears in the fold's
                 val.jsonl.zst or test.jsonl.zst
  3. loader      MultiTierLoader yields records for gold/silver/bronze under stable_cfg
                 and gold under anneal_cfg with STOICHEIA_DATA at the fold root
  4. eval        eval.intrinsic.held_out_records returns eval_n records

Usage:
    python data/verify_fold_shards.py --fold-src .../10-fold_split/fold_0 \
        --fold-root $STOICHEIA_DATA/folds/fold_0
Exits non-zero on any failure.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAIN_TIERS = {"pristine", "repaired", "bronze"}


def stream_ids(jsonl_zst, want_census=False):
    proc = subprocess.Popen(["zstdcat", str(jsonl_zst)], stdout=subprocess.PIPE, text=True)
    ids, n_rec, n_chars = set(), Counter(), Counter()
    for line in proc.stdout:
        rec = json.loads(line)
        ids.add(rec["id"])
        if want_census:
            n_rec[rec["tier"]] += 1
            n_chars[rec["tier"]] += len(rec.get("text", ""))
    if proc.wait() != 0:
        raise RuntimeError(f"zstdcat failed on {jsonl_zst}")
    return ids, n_rec, n_chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-src", required=True, help=".../10-fold_split/fold_k")
    ap.add_argument("--fold-root", required=True, help="$STOICHEIA_DATA/folds/fold_k")
    ap.add_argument("--eval-n", type=int, default=256)
    a = ap.parse_args()
    src, root = Path(a.fold_src), Path(a.fold_root)
    fails = []

    # ---- load shard indices ----
    shard_tier_rows, shard_tier_letters, shard_ids = Counter(), Counter(), set()
    for sd in ("v1_punct", "bronze_punct"):
        idx = pq.read_table(root / "shards" / sd / "index.parquet")
        tiers = idx.column("tier").to_numpy(zero_copy_only=False)
        lens = idx.column("length").to_numpy()
        offs = idx.column("offset").to_numpy()
        ids = idx.column("id").to_pylist()
        # offsets must be contiguous and match chars.bin size
        if idx.num_rows:
            ok = (offs[0] == 0 and np.all(offs[1:] == offs[:-1] + lens[:-1]))
            nbytes = (root / "shards" / sd / "chars.bin").stat().st_size
            if not ok or nbytes != int(offs[-1] + lens[-1]):
                fails.append(f"{sd}: offsets not contiguous or chars.bin size mismatch")
        for t in np.unique(tiers):
            m = tiers == t
            shard_tier_rows[str(t)] += int(m.sum())
            shard_tier_letters[str(t)] += int(lens[m].sum())
        shard_ids.update(i.split("#", 1)[0] for i in ids)
    # v1_punct must be pristine-first (holdout arithmetic parity with flagship)
    v1 = pq.read_table(root / "shards" / "v1_punct" / "index.parquet")
    tv = v1.column("tier").to_numpy(zero_copy_only=False)
    first_rep = np.flatnonzero(tv == "repaired")
    last_pri = np.flatnonzero(tv == "pristine")
    if len(first_rep) and len(last_pri) and first_rep[0] < last_pri[-1]:
        fails.append("v1_punct: not pristine-first ordered")

    # ---- 1. census vs source ----
    _, src_rec, src_chars = stream_ids(src / "train.jsonl.zst", want_census=True)
    print("tier        source_recs  shard_recs   source_chars   shard_letters")
    for t in sorted(src_rec):
        sr, hr = src_rec[t], shard_tier_rows.get(t, 0)
        flag = ""
        if t in TRAIN_TIERS:
            drop = 1 - hr / max(sr, 1)
            if drop > 0.05:
                flag = "  <-- FAIL >5% dropped"
                fails.append(f"census: tier {t} lost {drop:.1%} of records")
            # shard letters exclude punctuation/spaces, so only sanity-bound them
            if shard_tier_letters.get(t, 0) > src_chars[t]:
                fails.append(f"census: tier {t} shard letters exceed source chars")
        else:
            if hr:
                fails.append(f"census: non-training tier {t} present in shards")
        print(f"{t:<12}{sr:>11}  {hr:>10}  {src_chars[t]:>13}  {shard_tier_letters.get(t,0):>13}{flag}")

    # ---- 2. leakage ----
    for split in ("val", "test"):
        held_ids, _, _ = stream_ids(src / f"{split}.jsonl.zst")
        inter = shard_ids & held_ids
        if inter:
            fails.append(f"leakage: {len(inter)} train shard ids in {split} (e.g. {sorted(inter)[:5]})")
        print(f"leakage vs {split}: {len(inter)} overlaps / {len(held_ids)} {split} ids")

    # ---- 3. loader dry-run ----
    from train.data import MultiTierLoader, stable_cfg, anneal_cfg
    gdata = str(root)
    for label, cfg in (("stable", stable_cfg(gdata)), ("anneal", anneal_cfg(gdata))):
        ld = MultiTierLoader(cfg, rank=0, world_size=1)
        counts = {n: len(ld.elig[n]) for n in ld.names}
        want = {"gold", "silver", "bronze"} if label == "stable" else {"gold"}
        if set(ld.names) != want:
            fails.append(f"loader[{label}]: tiers {ld.names} != {sorted(want)}")
        recs = list(ld.records(5))
        if len(recs) != 5 or any(len(r["chars"]) == 0 for r in recs):
            fails.append(f"loader[{label}]: bad sample records")
        print(f"loader[{label}]: eligible {counts}, sampled {len(recs)} records ok")

    # ---- 4. eval holdout ----
    from eval.intrinsic import held_out_records
    recs = held_out_records(str(root / "shards" / "v1_punct"), a.eval_n)
    if len(recs) < a.eval_n:
        fails.append(f"eval: only {len(recs)}/{a.eval_n} held-out records")
    print(f"eval holdout: {len(recs)}/{a.eval_n} records")

    if fails:
        print("\nVERIFY FAILED:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("\nVERIFY OK")


if __name__ == "__main__":
    main()
