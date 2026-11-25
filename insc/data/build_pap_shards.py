"""Build memmap shards from documentary papyri TRAIN segments (TM digit rule:
3=test / 4=val never enter training data). Same plane format as Stoicheia
shards; tier='iphi' deliberately (interface compat: the finetune's TierSpec
filter and mix keys are reused unchanged), source='ddbdp'.

  python insc_data/build_pap_shards.py --out $INS_DATA/shards/pap_punct
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from papyri import load

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
    for r in load(split="train", min_len=a.min_len):
        for p in PLANES:
            bufs[p].append(r[p])
        rows["offset"].append(off); rows["length"].append(len(r["chars"]))
        rows["tier"].append("iphi"); rows["clean"].append(1.0)
        rows["source"].append("ddbdp"); rows["id"].append(f"{r['phi_id']}#{r['seg']}")
        rows["region_id"].append(r["region_id"]); rows["century_id"].append(r["century_id"])
        off += len(r["chars"])
    for p in PLANES:
        np.concatenate(bufs[p]).tofile(out / f"{p}.bin")
    pq.write_table(pa.table(rows), out / "index.parquet")
    (out / "stats.json").write_text(json.dumps(dict(records=len(rows["offset"]), letters=off)))
    print(f"papyri shards: {len(rows['offset']):,} records, {off/1e6:.1f}M letters -> {out}")


if __name__ == "__main__":
    main()
