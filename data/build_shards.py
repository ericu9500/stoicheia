"""Build memmap shards from the raw parquet corpus.

Output layout (--out):
  chars.bin boundary.bin dia.bin cap.bin   aligned uint8 planes, one entry per letter
  index.parquet                            per-record: offset, length, tier, clean, source, id
  stats.json                               merged normalization Stats

Records are laid out pristine-files-first, then repaired, preserving parquet order,
so downstream passes (dedup keeps the first copy seen) prefer clean text.
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.normalize import Stats, normalize_record

PLANES = ("chars", "boundary", "dia", "cap", "punct")


def process_file(args):
    pf, tmpdir = args
    pf, tmpdir = Path(pf), Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(pf)
    stats = Stats()
    bufs = {p: [] for p in PLANES}
    rows = {"offset": [], "length": [], "tier": [], "clean": [], "source": [], "id": []}
    off = 0
    texts = t.column("text").to_pylist()
    tiers = t.column("tier").to_pylist()
    cleans = t.column("clean").to_pylist()
    sources = t.column("source").to_pylist()
    ids = t.column("id").to_pylist()
    for text, tier, clean, source, rid in zip(texts, tiers, cleans, sources, ids):
        r = normalize_record(text, stats, with_punct=True)
        if r is None:
            continue
        chars, boundary, dia, cap, punct = r
        for p, a in zip(PLANES, r):
            bufs[p].append(a)
        rows["offset"].append(off)
        rows["length"].append(len(chars))
        rows["tier"].append(tier)
        rows["clean"].append(clean)
        rows["source"].append(source)
        rows["id"].append(rid)
        off += len(chars)
    for p in PLANES:
        np.concatenate(bufs[p]).tofile(tmpdir / f"{p}.bin")
    pq.write_table(pa.table(rows), tmpdir / "index.parquet")
    st = asdict(stats)
    st["stripped"] = dict(stats.stripped)
    st["archaic"] = dict(stats.archaic)
    st["other_marks"] = dict(stats.other_marks)
    (tmpdir / "stats.json").write_text(json.dumps(st))
    return str(pf.name), off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(raw.glob("pristine/*.parquet")) + sorted(raw.glob("repaired/*.parquet"))
    tmp = out / "tmp"
    jobs = [(str(f), str(tmp / f.stem)) for f in files]

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for name, n in ex.map(process_file, jobs):
            print(f"  {name}: {n/1e6:.1f}M letters", flush=True)

    # concatenate in canonical order
    total = Stats()
    tables, cum = [], 0
    for f in files:
        d = tmp / f.stem
        t = pq.read_table(d / "index.parquet")
        t = t.set_column(0, "offset", pa.array(t.column("offset").to_numpy() + cum))
        tables.append(t)
        cum += int(np.sum(t.column("length").to_numpy()))
        s = json.loads((d / "stats.json").read_text())
        st = Stats(**{k: s[k] for k in ("records_in", "records_kept",
                     "records_dropped_nongreek", "records_dropped_empty", "letters",
                     "words", "sentences", "mark_conflicts", "orphan_marks")})
        st.stripped = Counter({int(k): v for k, v in s["stripped"].items()})
        st.archaic = Counter(s["archaic"])
        st.other_marks = Counter(s["other_marks"])
        total.merge(st)
    for p in PLANES:
        with open(out / f"{p}.bin", "wb") as fo:
            for f in files:
                fo.write((tmp / f.stem / f"{p}.bin").read_bytes())
    pq.write_table(pa.concat_tables(tables), out / "index.parquet")
    st = asdict(total)
    st["stripped"] = {str(k): v for k, v in total.stripped.items()}
    st["archaic"] = dict(total.archaic)
    st["other_marks"] = dict(total.other_marks)
    (out / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"TOTAL letters: {cum/1e9:.3f}B  records kept: {total.records_kept}/{total.records_in}")
    print(f"words: {total.words/1e6:.1f}M  sentences: {total.sentences/1e6:.1f}M")


if __name__ == "__main__":
    main()
