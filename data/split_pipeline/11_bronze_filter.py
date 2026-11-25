#!/usr/bin/env python3
"""Stage 11: remove Greek-origin bronze from train (anti-paraphrase guard).

Bronze passages are machine back-translations of Latin works. Where the Latin
work is itself a translation of a GREEK work (Graeca miscellanea, Ptolemaeus
Latinus, Versiones latinae, the Bible, and any work by a Greek-writing author
per work/bronze_greek_origin_flags.json), the back-translation is a PARAPHRASE
of a real Greek text that may sit in val/test -- invisible to verbatim n-gram
matching but still memorization for conjecture evaluation. This stage rewrites
every fold's train.jsonl.zst dropping all bronze records whose cc_idno is
flagged, and records the counts.

Run AFTER stage 6; verify (stage 7) must be re-run afterwards for the final
certification (removal cannot introduce overlap, but we re-certify anyway).
"""
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import orjson
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))


def filter_fold(k):
    flags = json.load(open(os.path.join(ROOT, "work",
                                        "bronze_greek_origin_flags.json")))
    bad = {idno for idno, b in flags.items() if b}
    d = os.path.join(OUTDIR, "fold_%d" % k)
    src = os.path.join(d, "train.jsonl.zst")
    tmp = os.path.join(d, "train.filtered.jsonl.zst.tmp")
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=6)
    kept = dropped = dropped_chars = 0
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        writer = cctx.stream_writer(fout)
        with dctx.stream_reader(fin) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                if rec.get("tier") == "bronze":
                    idno = rec["id"].split(":")[0]
                    if idno in bad:
                        dropped += 1
                        dropped_chars += len(rec.get("text", ""))
                        continue
                kept += 1
                writer.write(line.encode())
        writer.close()
    os.replace(tmp, src)
    return k, kept, dropped, dropped_chars


def main():
    stats = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        for k, kept, dropped, dchars in ex.map(filter_fold, range(10)):
            stats["fold_%d" % k] = {"kept": kept, "dropped_bronze": dropped,
                                    "dropped_Mchars": round(dchars / 1e6, 1)}
            print("fold %d: kept %d, dropped %d bronze recs (%.1f Mchars)"
                  % (k, kept, dropped, dchars / 1e6), flush=True)
    with open(os.path.join(ROOT, "work", "stage11_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
