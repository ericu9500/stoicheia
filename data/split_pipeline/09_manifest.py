#!/usr/bin/env python3
"""Stage 9: per-fold manifests of the TEST split.

For every fold writes:
  fold_k/test_manifest.tsv  one row per test record:
      id, kind, source, group, author_title, nchars, preview
  fold_k/test_works.tsv     aggregated by group (volume/work/TM/PHI):
      group, kind, source, records, chars, author_title, preview
and TEST_MANIFEST.md at the root with per-fold summaries.

"group" is the same work/volume granularity the pipeline used for splitting:
oga tlgXXXX.tlgYYY, catholic/greek_pd/ia/pg/gutenberg volume prefix,
TM:<number> for papyri, PHI:<id> for inscriptions. Texts listed here are
guaranteed (see verify_report) to share no 8-word sequence and no sentence
with the fold's train (or val) set -- safe for reconstruction evaluation.
"""
import io
import os
import re
import sys
from collections import defaultdict

import orjson
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))
OGA_XML = os.path.expandvars("$STOICHEIA_DATA/raw/oga/"
           "opera_graeca_adnotata_v0.2.0/work_chronology/texts/"
           "chronology_greek_works.xml")
DDBDP_JSONL = os.path.expandvars("$STOICHEIA_DATA/clean/ddbdp.jsonl")
PAPYRI_TM_JSONL = os.path.expandvars("$STOICHEIA_DATA/data/papyri_clean.jsonl")


def load_oga_names():
    names = {}
    if not os.path.exists(OGA_XML):
        return names
    txt = io.open(OGA_XML, encoding="utf-8").read()
    for m in re.finditer(r"<record>(.*?)</record>", txt, re.S):
        blk = m.group(1)
        def g(tag):
            mm = re.search("<%s>(.*?)</%s>" % (tag, tag), blk, re.S)
            return mm.group(1).strip() if mm else ""
        urn = g("urn_cts")
        if urn:
            names[urn] = "%s — %s" % (g("author"), g("title_labels"))
    return names


def load_ddbdp_tm():
    tm_by_base = {}
    with open(PAPYRI_TM_JSONL, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            tm_by_base[r["file"]] = str(r["TM"])
    id2tm = {}
    with open(DDBDP_JSONL, "rb") as f:
        for line in f:
            r = orjson.loads(line)
            tm = tm_by_base.get(r["file"].rsplit("/", 1)[-1])
            if tm:
                id2tm[r["id"]] = tm
    return id2tm


def read_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    yield orjson.loads(line)


def clean(s, n=95):
    return re.sub(r"\s+", " ", s or "").strip()[:n]


def main():
    oga = load_oga_names()
    id2tm = load_ddbdp_tm()
    print("oga names: %d, ddbdp->TM: %d" % (len(oga), len(id2tm)))

    md = ["# Test-split manifests", "",
          "One row per test record in `fold_k/test_manifest.tsv`; aggregated",
          "by work/volume/document in `fold_k/test_works.tsv`.",
          "Every text listed is verified to share no 8-word sequence and no",
          "complete sentence with that fold's train and val sets (after",
          "orthographic normalization) — safe targets for text-reconstruction",
          "evaluation of a model trained on the same fold's train set.", ""]

    for k in range(10):
        d = os.path.join(OUTDIR, "fold_%d" % k)
        rows = []
        for r in read_zst(os.path.join(d, "test.jsonl.zst")):
            rid = r["id"]
            src = r.get("source", "")
            if r.get("tier") == "inscriptions":
                kind = "inscription"
                group = "PHI:%s" % r.get("PHI_ID")
                name = clean(r.get("main_region", "") or "", 40)
                text = r.get("with_diacritics") or r.get("edition") or \
                    r.get("ithaca_text") or ""
            else:
                text = r.get("text", "")
                if src == "dclp":
                    kind = "papyrus"
                    group = "TM:" + rid.split("_")[0]
                    name = ""
                elif src == "ddbdp":
                    kind = "papyrus"
                    tm = id2tm.get(rid)
                    group = "TM:%s" % tm if tm else "ddbdp:" + rid
                    name = rid
                else:
                    kind = "literary"
                    if src == "oga":
                        group = ".".join(rid.split(".")[:2])
                        name = oga.get(group, "")
                    elif "#" in rid:
                        group = rid.split("#")[0]
                        name = group if src == "catholic" else ""
                    else:
                        group = rid
                        name = ""
                    group = src + ":" + group
            rows.append((rid, kind, src, group, name, len(text), clean(text)))

        with io.open(os.path.join(d, "test_manifest.tsv"), "w",
                     encoding="utf-8") as f:
            f.write("id\tkind\tsource\tgroup\tauthor_title\tnchars\tpreview\n")
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")

        groups = defaultdict(lambda: [0, 0, "", "", "", ""])
        for rid, kind, src, group, name, nch, prev in rows:
            g = groups[group]
            g[0] += 1
            g[1] += nch
            if nch >= len(g[5]):
                g[2], g[3], g[4], g[5] = kind, src, name or g[4], prev
            elif name and not g[4]:
                g[4] = name
        with io.open(os.path.join(d, "test_works.tsv"), "w",
                     encoding="utf-8") as f:
            f.write("group\tkind\tsource\trecords\tchars\tauthor_title\tpreview\n")
            for group, g in sorted(groups.items(), key=lambda x: -x[1][1]):
                f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\n" %
                        (group, g[2], g[3], g[0], g[1], g[4], g[5]))

        by_kind = defaultdict(lambda: [0, 0])
        for _, kind, *_rest in rows:
            pass
        for rid, kind, src, group, name, nch, prev in rows:
            by_kind[kind][0] += 1
            by_kind[kind][1] += nch
        ngroups = len(groups)
        md += ["## fold %d — %d records, %d works/documents" %
               (k, len(rows), ngroups), ""]
        md += ["| kind | records | Mchars |", "|---|---|---|"]
        for kind in ("literary", "papyrus", "inscription"):
            c = by_kind.get(kind, [0, 0])
            md.append("| %s | %d | %.1f |" % (kind, c[0], c[1] / 1e6))
        md += ["", "Largest test works:", ""]
        top = sorted(groups.items(), key=lambda x: -x[1][1])[:12]
        for group, g in top:
            label = g[4] or g[5][:60]
            md.append("- `%s` (%d recs, %.2f Mchars) %s" %
                      (group, g[0], g[1] / 1e6, label))
        md.append("")
        print("fold %d: %d records, %d groups" % (k, len(rows), ngroups))

    with io.open(os.path.join(OUTDIR, "TEST_MANIFEST.md"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(md))
    print("wrote", os.path.join(OUTDIR, "TEST_MANIFEST.md"))


if __name__ == "__main__":
    main()
