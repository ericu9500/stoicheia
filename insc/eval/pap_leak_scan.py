"""Contamination scan: which papyrus (DDbDP) val/test segments occur VERBATIM in the
pretraining corpus? Same method as leak_scan.py (inscriptions): searches the raw
letter-id planes (accentless — catches matches regardless of diacritics/spacing/
punctuation differences). Writes a JSON of contaminated (tm_id, seg) keys for
eval-side exclusion.

  python insc/eval/pap_leak_scan.py --split val --out $INS_DATA/contaminated_pap_val.json
"""
from __future__ import annotations

import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from papyri import load as load_iphi

PLANES = [
    os.path.expandvars("$STOICHEIA_DATA/shards/v1_punct/chars.bin"),
    os.path.expandvars("$STOICHEIA_DATA/shards/bronze_punct/chars.bin"),
]
_corpora = None


def _init():
    global _corpora
    _corpora = [Path(p).read_bytes() for p in PLANES]


def _probe(args):
    key, q = args
    for ci, c in enumerate(_corpora):
        if c.find(q) >= 0:
            return (key, ci)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--window", type=int, default=48, help="probe length (letters)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planes", default=None,
                    help="comma-separated chars.bin paths to scan (default: flagship shards)")
    a = ap.parse_args()
    if a.planes:
        PLANES[:] = a.planes.split(",")

    recs = [r for r in load_iphi(split=a.split, min_len=50) if len(r["chars"]) <= 1500]
    queries = []
    for r in recs:
        ch = np.asarray(r["chars"], np.uint8)
        w = a.window
        # probe up to three windows (25%/50%/75% anchors) — a duplicate anywhere in the
        # segment should trip the scan, not only one at its middle
        anchors = sorted({len(ch) // 4, len(ch) // 2, (3 * len(ch)) // 4})
        for mid in anchors:
            q = ch[max(0, mid - w // 2): max(0, mid - w // 2) + w].tobytes()
            if len(q) == w:
                queries.append(((str(r["phi_id"]), int(r["seg"])), q))
    # positive control: a window copied from the corpus itself must hit
    corpus0 = np.memmap(PLANES[0], dtype=np.uint8, mode="r")
    queries.append((("POSITIVE_CONTROL", 0), np.asarray(corpus0[10_000_000:10_000_000 + a.window]).tobytes()))

    with Pool(a.workers, initializer=_init) as pool:
        hits = [h for h in pool.map(_probe, queries, chunksize=16) if h]

    ctl = [h for h in hits if h[0][0] == "POSITIVE_CONTROL"]
    real = sorted({h[0] for h in hits if h[0][0] != "POSITIVE_CONTROL"})
    assert ctl, "positive control FAILED — scan is broken"
    n_seg = len({k for k, _ in queries if k[0] != "POSITIVE_CONTROL"})
    print(f"split={a.split}: {len(real)}/{n_seg} segments found verbatim in "
          f"pretraining planes ({100*len(real)/n_seg:.2f}%)")
    Path(a.out).write_text(json.dumps(dict(
        split=a.split, window=a.window, n_scanned=n_seg,
        contaminated=[list(k) for k in real])))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
