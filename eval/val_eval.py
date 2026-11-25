"""Offline intrinsic eval on a fold's REAL val split (the unseen-works 10%).

The in-training eval (eval/intrinsic via train.py) scores held-out records from
the training distribution — segments of works whose other segments ARE trained
on. This script scores the same metrics on the fold's val bucket: whole works
the model has never seen any part of. Run it on any checkpoint, any time; it
appends to <out_dir>/val_eval.jsonl and never touches training state.

Usage (inside the training container, 1 GPU):
    python -m eval.val_eval --ckpt $STOICHEIA_DATA/runs/stoicheia_fold_0/best.pt \
        --val-shards $STOICHEIA_DATA/folds/fold_0/val_shards/v1_punct [--n 1024]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch

from eval.intrinsic import evaluate, load_model


def val_records(shards, n, seed=1234):
    """Sample n pristine val records (64..4096 chars) — ALL are unseen works."""
    import pyarrow.parquet as pq
    d = Path(shards)
    idx = pq.read_table(d / "index.parquet")
    offs = idx.column("offset").to_numpy(); lens = idx.column("length").to_numpy()
    tier = idx.column("tier").to_numpy(zero_copy_only=False)
    planes = {p: np.memmap(d / f"{p}.bin", dtype=np.uint8, mode="r")
              for p in ("chars", "boundary", "dia", "cap", "punct")}
    ok = np.flatnonzero((tier == "pristine") & (lens >= 64) & (lens <= 4096))
    rng = np.random.default_rng(seed)
    pick = rng.choice(ok, size=min(n, len(ok)), replace=False)
    recs = []
    for i in pick:
        o, l = int(offs[i]), int(lens[i])
        recs.append({p: np.array(a[o:o+l]) for p, a in planes.items()})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-shards", required=True)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", default=None,
                    help="output jsonl (default: <ckpt dir>/val_eval.jsonl)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.ckpt, device)
    sd_step = torch.load(a.ckpt, map_location="cpu").get("step", -1)
    recs = val_records(a.val_shards, a.n)
    m = evaluate(model, recs, device)
    m.update(step=int(sd_step), ckpt=Path(a.ckpt).name, n_records=len(recs),
             split="val")
    out = Path(a.out) if a.out else Path(a.ckpt).parent / "val_eval.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(m) + "\n")
    print("VAL_EVAL", json.dumps(m), flush=True)


if __name__ == "__main__":
    main()
