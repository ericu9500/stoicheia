"""Build per-fold memmap shards from a 10-fold_split fold's train.jsonl.zst.

Same output format/planes as build_shards.py / build_bronze.py so the training loader
works unchanged when STOICHEIA_DATA points at the fold root:

    <out>/v1_punct/       tier in {pristine, repaired}   (pristine rows first, then
                          repaired — mirrors the flagship's canonical order so the
                          idx%HOLDOUT_MOD holdout and eval/intrinsic sampling behave
                          identically)
    <out>/bronze_punct/   tier == bronze

Records with tier == inscriptions (or anything else) are skipped and counted — the
flagship recipe never trains on inscriptions. Record ids (including the #segN suffixes
of excised-and-restitched train segments) are kept verbatim.

Usage:
    python data/build_fold_shards.py --jsonl .../fold_0/train.jsonl.zst \
        --out $STOICHEIA_DATA/folds/fold_0/shards --workers 16
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys
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
TIER2TARGET = {"pristine": "pri", "repaired": "rep", "bronze": "brz"}
CLEAN = {"pristine": 1.0, "repaired": 0.75, "bronze": 0.5}
INDEX_SCHEMA = pa.schema([("offset", pa.int64()), ("length", pa.int64()),
                          ("tier", pa.string()), ("clean", pa.float64()),
                          ("source", pa.string()), ("id", pa.string())])


def _dump_stats(stats: Stats, path: Path):
    st = asdict(stats)
    st["stripped"] = dict(stats.stripped)
    st["archaic"] = dict(stats.archaic)
    st["other_marks"] = dict(stats.other_marks)
    path.write_text(json.dumps(st))


def _load_stats(path: Path) -> Stats:
    s = json.loads(path.read_text())
    st = Stats(**{k: s[k] for k in ("records_in", "records_kept",
               "records_dropped_nongreek", "records_dropped_empty", "letters",
               "words", "sentences", "mark_conflicts", "orphan_marks")})
    st.stripped = Counter({int(k): v for k, v in s["stripped"].items()})
    st.archaic = Counter(s["archaic"])
    st.other_marks = Counter(s["other_marks"])
    return st


def process_chunk(args):
    chunk_file, tmpdir, cid = args
    tmpdir = Path(tmpdir)
    stats = {t: Stats() for t in ("pri", "rep", "brz")}
    bufs = {t: {p: [] for p in PLANES} for t in ("pri", "rep", "brz")}
    rows = {t: {"offset": [], "length": [], "tier": [], "clean": [], "source": [], "id": []}
            for t in ("pri", "rep", "brz")}
    off = {t: 0 for t in ("pri", "rep", "brz")}
    skipped = Counter()
    with open(chunk_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                skipped["unparseable"] += 1
                continue
            tier = rec.get("tier", "")
            t = TIER2TARGET.get(tier)
            if t is None:
                skipped[tier or "missing_tier"] += 1
                continue
            r = normalize_record(rec.get("text", ""), stats[t], with_punct=True)
            if r is None:
                continue
            for p, a in zip(PLANES, r):
                bufs[t][p].append(a)
            rows[t]["offset"].append(off[t])
            rows[t]["length"].append(len(r[0]))
            rows[t]["tier"].append(tier)
            rows[t]["clean"].append(CLEAN[tier])
            rows[t]["source"].append(rec.get("source", ""))
            rows[t]["id"].append(rec.get("id", ""))
            off[t] += len(r[0])
    for t in ("pri", "rep", "brz"):
        d = tmpdir / t
        d.mkdir(parents=True, exist_ok=True)
        for p in PLANES:
            if bufs[t][p]:
                np.concatenate(bufs[t][p]).tofile(d / f"{p}.bin")
            else:
                np.array([], np.uint8).tofile(d / f"{p}.bin")
        pq.write_table(pa.table(rows[t], schema=INDEX_SCHEMA), d / "index.parquet")
        _dump_stats(stats[t], d / "stats.json")
    (tmpdir / "skipped.json").write_text(json.dumps(dict(skipped)))
    return cid


def assemble(out_dir: Path, part_dirs: list[Path]):
    """Concatenate per-chunk target dirs (in the given order) into one shard dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = Stats()
    tables, cum = [], 0
    for d in part_dirs:
        t = pq.read_table(d / "index.parquet")
        if t.num_rows:
            t = t.set_column(0, "offset", pa.array(t.column("offset").to_numpy() + cum,
                                                   type=pa.int64()))
            tables.append(t)
            cum += int(np.sum(t.column("length").to_numpy()))
        total.merge(_load_stats(d / "stats.json"))
    for p in PLANES:
        with open(out_dir / f"{p}.bin", "wb") as fo:
            for d in part_dirs:
                fo.write((d / f"{p}.bin").read_bytes())
    pq.write_table(pa.concat_tables(tables) if tables else INDEX_SCHEMA.empty_table(),
                   out_dir / "index.parquet")
    st = asdict(total)
    st["stripped"] = {str(k): v for k, v in total.stripped.items()}
    st["archaic"] = dict(total.archaic)
    st["other_marks"] = dict(total.other_marks)
    (out_dir / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
    return cum, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="fold_k/train.jsonl.zst")
    ap.add_argument("--out", required=True, help="fold root shards dir (gets v1_punct/, bronze_punct/)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=40000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "chunks").mkdir(parents=True)

    # stream-decompress and split into line-chunk files (keeps memory flat)
    proc = subprocess.Popen(["zstdcat", a.jsonl], stdout=subprocess.PIPE, text=True)
    chunks, buf, cid = [], [], 0
    for line in proc.stdout:
        buf.append(line)
        if len(buf) >= a.chunk:
            cf = tmp / "chunks" / f"c{cid:04d}.jsonl"
            cf.write_text("".join(buf))
            chunks.append((str(cf), str(tmp / f"c{cid:04d}"), cid))
            buf, cid = [], cid + 1
    if buf:
        cf = tmp / "chunks" / f"c{cid:04d}.jsonl"
        cf.write_text("".join(buf))
        chunks.append((str(cf), str(tmp / f"c{cid:04d}"), cid))
    if proc.wait() != 0:
        raise RuntimeError(f"zstdcat failed on {a.jsonl}")
    print(f"{len(chunks)} chunks", flush=True)

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(process_chunk, chunks))
    order = [Path(c[1]) for c in chunks]

    # canonical order: all pristine parts, then all repaired parts (mirrors build_shards.py)
    lit_letters, lit_stats = assemble(out / "v1_punct",
                                      [d / "pri" for d in order] + [d / "rep" for d in order])
    brz_letters, brz_stats = assemble(out / "bronze_punct", [d / "brz" for d in order])

    skipped = Counter()
    for d in order:
        skipped.update(json.loads((d / "skipped.json").read_text()))
    prov = {
        "source_jsonl": str(Path(a.jsonl).resolve()),
        "source_mtime": Path(a.jsonl).stat().st_mtime,
        "v1_punct": {"letters": lit_letters, "records_kept": lit_stats.records_kept,
                     "records_in": lit_stats.records_in},
        "bronze_punct": {"letters": brz_letters, "records_kept": brz_stats.records_kept,
                         "records_in": brz_stats.records_in},
        "skipped": dict(skipped),
    }
    (out.parent / "provenance.json").write_text(json.dumps(prov, indent=2))
    shutil.rmtree(tmp)
    print(f"v1_punct letters: {lit_letters/1e9:.3f}B  kept: {lit_stats.records_kept}/{lit_stats.records_in}")
    print(f"bronze_punct letters: {brz_letters/1e9:.3f}B  kept: {brz_stats.records_kept}/{brz_stats.records_in}")
    print(f"skipped (non-training tiers): {dict(skipped)}")


if __name__ == "__main__":
    main()
