"""Build memmap shards from bronze.jsonl (synthetic translated Greek).

Same output format/planes as build_shards.py so the training loader unions gold/silver/bronze
transparently. tier='bronze', clean=0.5 nominal (synthetic). The <95%-Greek filter in
normalize drops encoding-garbage / Latin-contaminated generations automatically.
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


def process_chunk(args):
    lines, tmpdir, cid = args
    tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    bufs = {p: [] for p in PLANES}
    rows = {"offset": [], "length": [], "tier": [], "clean": [], "source": [], "id": []}
    off = 0
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        r = normalize_record(rec.get("text", ""), stats, with_punct=True)
        if r is None:
            continue
        for p, a in zip(PLANES, r):
            bufs[p].append(a)
        rows["offset"].append(off)
        rows["length"].append(len(r[0]))
        rows["tier"].append("bronze")
        rows["clean"].append(0.5)
        rows["source"].append(rec.get("corpus", "bronze"))
        rows["id"].append(rec.get("id", ""))
        off += len(r[0])
    for p in PLANES:
        if bufs[p]:
            np.concatenate(bufs[p]).tofile(tmpdir / f"{p}.bin")
        else:
            np.array([], np.uint8).tofile(tmpdir / f"{p}.bin")
    pq.write_table(pa.table(rows), tmpdir / "index.parquet")
    st = asdict(stats)
    st["stripped"] = dict(stats.stripped); st["archaic"] = dict(stats.archaic)
    st["other_marks"] = dict(stats.other_marks)
    (tmpdir / "stats.json").write_text(json.dumps(st))
    return cid, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=40000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"

    # split file into line chunks
    chunks, buf, cid = [], [], 0
    with open(a.jsonl) as f:
        for line in f:
            buf.append(line)
            if len(buf) >= a.chunk:
                chunks.append((buf, str(tmp / f"c{cid:04d}"), cid)); buf = []; cid += 1
        if buf:
            chunks.append((buf, str(tmp / f"c{cid:04d}"), cid))
    print(f"{len(chunks)} chunks")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        done = sorted(ex.map(process_chunk, chunks))
    order = [c[2] for c in chunks]

    total = Stats(); tables, cum = [], 0
    for cid in order:
        d = tmp / f"c{cid:04d}"
        t = pq.read_table(d / "index.parquet")
        if t.num_rows:
            t = t.set_column(0, "offset", pa.array(t.column("offset").to_numpy() + cum))
            tables.append(t)
            cum += int(np.sum(t.column("length").to_numpy()))
        s = json.loads((d / "stats.json").read_text())
        st = Stats(**{k: s[k] for k in ("records_in", "records_kept",
                     "records_dropped_nongreek", "records_dropped_empty", "letters",
                     "words", "sentences", "mark_conflicts", "orphan_marks")})
        st.stripped = Counter({int(k): v for k, v in s["stripped"].items()})
        total.merge(st)
    for p in PLANES:
        with open(out / f"{p}.bin", "wb") as fo:
            for cid in order:
                fo.write((tmp / f"c{cid:04d}" / f"{p}.bin").read_bytes())
    pq.write_table(pa.concat_tables(tables), out / "index.parquet")
    st = asdict(total); st["stripped"] = {str(k): v for k, v in total.stripped.items()}
    st["archaic"] = dict(total.archaic); st["other_marks"] = dict(total.other_marks)
    (out / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    print(f"bronze letters: {cum/1e9:.3f}B  kept: {total.records_kept}/{total.records_in}  "
          f"words: {total.words/1e6:.1f}M")


if __name__ == "__main__":
    main()
