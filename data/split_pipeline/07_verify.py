#!/usr/bin/env python3
"""Stage 7: INDEPENDENT watertightness verification of the materialized folds.

Deliberately does NOT import common.py. Re-implements normalization from
scratch (codepoint table built from unicodedata, not the regex module),
re-tokenizes, and re-hashes (blake2b, not xxhash). For each fold it asserts:

  V1: no word-8-gram of any train record appears in any val/test record
  V2: no full sentence skeleton of any train record equals a val/test sentence
  V3: same two checks between val and test (val must be clean of test)

The spec constants shared with the pipeline (8-gram size, sentence terminator
characters, sigma folding) are re-declared here as documented constants.
"""
import glob
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import orjson
import zstandard as zstd

OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))
NGRAM = 8  # == pipeline spec (user-approved strictness: shared 8-gram = leak)
SENT_RE = re.compile("[.;!?:\u00b7\u0387\u037e]+|\\n\\s*\\n")
INSCR_REAL = ["edition", "with_diacritics", "without_diacritics", "ithaca_text"]

# ---- independent normalization: one translate table over all codepoints ----
_SIGMA_MAP = {"ς": "σ", "ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϰ": "κ"}


def _build_table():
    table = {}
    for cp in range(0x30000):
        ch = chr(cp)
        if unicodedata.combining(ch):
            table[cp] = None                      # strip diacritics
            continue
        lo = ch.lower()
        keep = []
        for c in lo:                               # lower() may expand
            c = _SIGMA_MAP.get(c, c)
            # Greek script letters: basic block minus Coptic 03E2-03EF,
            # plus Greek Extended (should not survive NFD, kept for safety)
            o = ord(c)
            is_greek = (0x0370 <= o <= 0x03FF and not 0x03E2 <= o <= 0x03EF) \
                or (0x1F00 <= o <= 0x1FFF)
            if is_greek and unicodedata.category(c).startswith("L"):
                keep.append(c)
            else:
                keep.append(" ")
        table[cp] = "".join(keep)
    return table


TABLE = _build_table()


def norm_words(text):
    return unicodedata.normalize("NFD", text).translate(TABLE).split()


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def unit_texts(rec):
    if "text" in rec:
        yield rec["text"]
    else:
        for fld in INSCR_REAL:
            v = rec.get(fld)
            if v and str(v).strip():
                yield str(v)


def rec_hashes(rec):
    """(gram_hashes, sentence_hashes) for one output record."""
    grams, sents = [], []
    for text in unit_texts(rec):
        stream = []
        for part in SENT_RE.split(unicodedata.normalize("NFD", text)):
            if part is None or not part:
                continue
            w = norm_words(part)
            if w:
                sents.append(h(" ".join(w)))
                stream.extend(w)
        for i in range(len(stream) - NGRAM + 1):
            grams.append(h(" ".join(stream[i:i + NGRAM])))
    return grams, sents


def read_jsonl_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    yield orjson.loads(line)


G = {}


def check_batch(lines):
    gk, sk = G["gk"], G["sk"]
    viol = []
    n_g = n_s = 0
    for raw in lines:
        rec = orjson.loads(raw)
        grams, sents = rec_hashes(rec)
        for name, q, keys in (("gram", grams, gk), ("sent", sents, sk)):
            if not q:
                continue
            qa = np.array(q, dtype=np.uint64)
            pos = np.searchsorted(keys, qa)
            pos[pos >= len(keys)] = len(keys) - 1
            nhit = int((keys[pos] == qa).sum())
            if nhit:
                if name == "gram":
                    n_g += nhit
                else:
                    n_s += nhit
                if len(viol) < 5:
                    viol.append({"id": rec.get("id"), "kind": name,
                                 "hits": nhit})
    return n_g, n_s, viol


def build_reference(paths):
    grams, sents = [], []
    for p in paths:
        for rec in read_jsonl_zst(p):
            g, s = rec_hashes(rec)
            grams.extend(g)
            sents.extend(s)
    gk = np.unique(np.array(grams, dtype=np.uint64))
    sk = np.unique(np.array(sents, dtype=np.uint64))
    return gk, sk


def scan(path, workers, batch=4000):
    total_g = total_s = 0
    samples = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        buf = []
        dctx = zstd.ZstdDecompressor()
        with open(path, "rb") as f:
            with dctx.stream_reader(f) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    if line.strip():
                        buf.append(line)
                    if len(buf) >= batch:
                        futs.append(ex.submit(check_batch, buf))
                        buf = []
                        if len(futs) > workers * 4:   # bound memory
                            for fu in futs:
                                g, s, v = fu.result()
                                total_g += g
                                total_s += s
                                samples.extend(v)
                            futs = []
        if buf:
            futs.append(ex.submit(check_batch, buf))
        for fu in futs:
            g, s, v = fu.result()
            total_g += g
            total_s += s
            samples.extend(v)
    return total_g, total_s, samples[:10]


def main():
    folds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                              else range(10))]
    workers = max(4, os.cpu_count() - 8)
    report = {}
    for k in folds:
        d = os.path.join(OUTDIR, "fold_%d" % k)
        print("fold %d: building val+test reference..." % k, flush=True)
        gk, sk = build_reference([os.path.join(d, "val.jsonl.zst"),
                                  os.path.join(d, "test.jsonl.zst")])
        G["gk"], G["sk"] = gk, sk
        print("  reference: %d grams, %d sentences" % (len(gk), len(sk)),
              flush=True)
        tg, ts, sv = scan(os.path.join(d, "train.jsonl.zst"), workers)
        # val vs test
        gk2, sk2 = build_reference([os.path.join(d, "test.jsonl.zst")])
        G["gk"], G["sk"] = gk2, sk2
        vg, vs, vv = scan(os.path.join(d, "val.jsonl.zst"), workers)
        report["fold_%d" % k] = {
            "train_vs_valtest": {"gram_hits": tg, "sent_hits": ts,
                                 "samples": sv},
            "val_vs_test": {"gram_hits": vg, "sent_hits": vs, "samples": vv},
            "PASS": tg == 0 and ts == 0 and vg == 0 and vs == 0,
        }
        print("fold %d: train-vs-valtest grams=%d sents=%d | val-vs-test "
              "grams=%d sents=%d -> %s" %
              (k, tg, ts, vg, vs,
               "PASS" if report["fold_%d" % k]["PASS"] else "FAIL"), flush=True)
    outp = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "work", "verify_report.json")
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    if not all(v["PASS"] for v in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
