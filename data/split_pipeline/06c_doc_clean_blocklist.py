#!/usr/bin/env python3
"""Stage 6c (documentary-clean variant): wholesale exclusion of known documentary
SOURCEBOOK works from the doc-clean corpus.

Stage 5b's exact/bag/8-gram sentence matching plus its document-level MinHash-LSH
safety net (stage 4c) together drop the large majority of documentary echoes hiding
in the literary/bronze corpus. But century-old critical editions (Dittenberger's
Sylloge Inscriptionum Graecarum, Schwyzer's Dialectorum Graecarum Exempla Epigraphica
Potiora, Cagnat's Inscriptiones Graecae ad Res Romanas Pertinentes, and similar
epigraphic/papyrological corpora catalogued as ordinary "literary" books in the IA
tier) diverge from PHI/TM's modern editions -- different restorations, different line
divisions, a century of scholarship apart -- enough that some passages survive both
automated checks while still being, in substance, reproductions of documentary text.

For these specifically-identified sourcebook works, exclude EVERY segment regardless
of match status: "documentary content" here is not a per-sentence property but a
property of the WORK (its entire purpose is to catalogue inscriptions/papyri), so
partial/automated matching is the wrong tool -- wholesale exclusion is.

Run AFTER stage 6b (and after the bronze Greek-origin filter). Extend
SOURCEBOOK_WORK_PREFIXES as further such volumes are identified (e.g. via an
archive.org title audit for epigraphic/papyrological corpora).
"""
import io
import json
import os

import orjson
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("DOC_OUTDIR",
                        "$STOICHEIA_DATA"))

# IA/archive.org work ids identified as epigraphic/papyrological SOURCEBOOKS (the
# entire volume is a corpus of inscriptions or papyri, not incidental quotation).
# Traced via leak_scan contamination hits against fold_0/fold_pap eval segments.
SOURCEBOOK_WORK_PREFIXES = (
    "syllogeinscript02dittgoog",    # Dittenberger, Sylloge Inscriptionum Graecarum
    "syllogeinscript01dittgoog",
    "dialectorumgraec00schw",       # Schwyzer, Dialectorum Graecarum Exempla Epigraphica Potiora
    "inscriptionesarg11unse",       # IG IV (Argolid) sourcebook
    "inscriptionesgra04cagnuoft",   # Cagnat, Inscriptiones Graecae ad Res Romanas Pertinentes
    "inscriptionesgra01cagnuoft",
    "inscriptionesins0000unse",     # Inscriptiones Insularum sourcebook
    "papersathens03ameruoft",       # ASCSA Papers (epigraphic reports)
)


def main():
    src = os.path.join(OUTDIR, "train.jsonl.zst")
    tmp = src + ".tmp"
    dctx, cctx = zstd.ZstdDecompressor(), zstd.ZstdCompressor(level=6)
    kept = dropped = dropped_chars = 0
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        w = cctx.stream_writer(fout)
        with dctx.stream_reader(fin) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = orjson.loads(line)
                rid = rec.get("id", "")
                if any(rid.startswith(p) for p in SOURCEBOOK_WORK_PREFIXES):
                    dropped += 1
                    dropped_chars += len(rec.get("text", ""))
                    continue
                kept += 1
                w.write(line.encode())
        w.close()
    os.replace(tmp, src)
    stats = dict(kept=kept, dropped=dropped, dropped_chars=dropped_chars,
                prefixes=list(SOURCEBOOK_WORK_PREFIXES))
    with open(os.path.join(ROOT, "work", "doc_clean", "stage6c_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"kept {kept}  dropped {dropped} sourcebook-work segments "
          f"({dropped_chars/1e6:.1f}M chars)")


if __name__ == "__main__":
    main()
