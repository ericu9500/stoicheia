"""Build memmap shards from I.PHI TRAIN segments (val/test never enter training data).

Same plane format as Stoicheia shards so MultiTierLoader can mix iphi with the
pretraining tiers transparently. tier='iphi', clean=1.0.

  python insc_data/build_shards.py --out $INS_DATA/shards/iphi_punct
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iphi import load

PLANES = ("chars", "boundary", "dia", "cap", "punct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=32)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    bufs = {p: [] for p in PLANES}
    rows = {"offset": [], "length": [], "tier": [], "clean": [], "source": [], "id": [],
            "region_id": [], "century_id": []}
    off = 0
    # real edition text (tier iphi) + synthetic paraphrase columns (tier iphi_syn).
    # ONLY train-split inscriptions in both cases: the synthetic columns of val/test
    # items paraphrase the held-out answers — using them would leak the eval.
    for field, tier in (("with_diacritics", "iphi"), ("synthetic", "iphi_syn"),
                        ("synthetic_2", "iphi_syn")):
        for r in load(split="train", min_len=a.min_len, field=field):
            for p in PLANES:
                bufs[p].append(r[p])
            rows["offset"].append(off); rows["length"].append(len(r["chars"]))
            rows["tier"].append(tier); rows["clean"].append(1.0)
            rows["source"].append(field); rows["id"].append(f"{r['phi_id']}#{r['seg']}")
            rows["region_id"].append(r["region_id"]); rows["century_id"].append(r["century_id"])
            off += len(r["chars"])
    for p in PLANES:
        np.concatenate(bufs[p]).tofile(out / f"{p}.bin")
    pq.write_table(pa.table(rows), out / "index.parquet")
    (out / "stats.json").write_text(json.dumps(dict(records=len(rows["offset"]), letters=off)))
    print(f"iphi shards: {len(rows['offset']):,} records, {off/1e6:.1f}M letters -> {out}")


if __name__ == "__main__":
    main()
