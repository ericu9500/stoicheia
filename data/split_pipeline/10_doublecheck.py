#!/usr/bin/env python3
"""Stage 10: adversarial double-check of the materialized folds.

Beyond re-running the independent verifier (stage 7), this asserts:

  D1 MUTATION TEST -- the verifier is not vacuous. Records sampled from
     fold_0 test, plus orthographically mutated copies (diacritics stripped,
     punctuation replaced, sigma variants), MUST be flagged against the test
     reference; genuine train records must not.
  D2 DIGIT-RULE COMPLIANCE -- in every fold: every test inscription PHI id
     ends in 3, every val one in 4; every test/val papyrus TM ends in 3/4;
     no train record (any tier) carries a mapped TM/PHI ending in 3 or 4.
  D3 ID-DISJOINTNESS -- within each fold, no base record id appears in more
     than one of train/val/test.

Writes work/doublecheck_report.json; exits non-zero on any failure.
"""
import importlib.util
import io
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import orjson
import zstandard as zstd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))

spec = importlib.util.spec_from_file_location("verify", os.path.join(HERE, "07_verify.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

DDBDP_JSONL = os.path.expandvars("$STOICHEIA_DATA/clean/ddbdp.jsonl")
PAPYRI_TM_JSONL = os.path.expandvars("$STOICHEIA_DATA/data/papyri_clean.jsonl")


def read_zst(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    yield line


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


def mutate(text):
    """Plausible 'other edition': strip diacritics, change punctuation/sigmas."""
    import unicodedata
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.replace(".", "·").replace(",", "").replace("ς", "σ")


def d1_mutation_test():
    d = os.path.join(OUTDIR, "fold_0")
    gk, sk = verify.build_reference([os.path.join(d, "test.jsonl.zst")])
    verify.G["gk"], verify.G["sk"] = gk, sk
    test_lines, train_lines = [], []
    for line in read_zst(os.path.join(d, "test.jsonl.zst")):
        test_lines.append(line)
        if len(test_lines) >= 400:
            break
    for line in read_zst(os.path.join(d, "train.jsonl.zst")):
        train_lines.append(line)
        if len(train_lines) >= 4000:
            break
    g_plain, s_plain, _ = verify.check_batch(test_lines)
    mutated = []
    for line in test_lines:
        r = orjson.loads(line)
        for fld in ("text", "with_diacritics", "edition"):
            if isinstance(r.get(fld), str):
                r[fld] = mutate(r[fld])
        mutated.append(orjson.dumps(r).decode())
    g_mut, s_mut, _ = verify.check_batch(mutated)
    g_train, s_train, _ = verify.check_batch(train_lines)
    res = {"planted_verbatim": {"gram_hits": g_plain, "sent_hits": s_plain},
           "planted_mutated_edition": {"gram_hits": g_mut, "sent_hits": s_mut},
           "genuine_train_sample": {"gram_hits": g_train, "sent_hits": s_train},
           "PASS": g_plain > 0 and s_plain > 0 and g_mut > 0
                   and g_train == 0 and s_train == 0}
    return res


def check_fold(args):
    k, id2tm = args
    d = os.path.join(OUTDIR, "fold_%d" % k)
    want = {"test": "3", "val": "4"}
    bad_digit = []
    seen = {}
    dup_across = []
    for split in ("train", "val", "test"):
        for line in read_zst(os.path.join(d, split + ".jsonl.zst")):
            r = orjson.loads(line)
            rid = str(r["id"])
            base = rid.split("#")[0]
            if base.startswith("phi") and ":" in base:
                base = base.split(":")[0]
            if base.startswith("tlg") and base.count(".") >= 2:
                base = ".".join(base.split(".")[:2])   # oga work granularity
            prev = seen.get(base)
            if prev is not None and prev != split:
                if len(dup_across) < 5:
                    dup_across.append((base, prev, split))
            seen[base] = split
            # digit rule
            tm = None
            if r.get("tier") == "inscriptions":
                tm = str(r["PHI_ID"]) if "PHI_ID" in r else base.replace("phi", "")
            elif r.get("source") == "dclp":
                tm = base.split("_")[0]
            elif r.get("source") == "ddbdp":
                tm = id2tm.get(base)
            if tm is None:
                continue
            dig = tm.rstrip()[-1]
            if split in ("val", "test"):
                if dig != want[split] and len(bad_digit) < 5:
                    bad_digit.append((split, rid, tm))
            else:
                if dig in ("3", "4") and len(bad_digit) < 5:
                    bad_digit.append((split, rid, tm))
    n_dup = len(dup_across)
    return k, {"digit_violations": bad_digit, "cross_split_dup_ids": dup_across,
               "PASS": not bad_digit and not dup_across}


def main():
    report = {}
    print("D1: mutation test of the verifier...", flush=True)
    report["D1_mutation"] = d1_mutation_test()
    print(json.dumps(report["D1_mutation"], indent=2), flush=True)

    id2tm = load_ddbdp_tm()
    print("D2+D3: digit-rule + id-disjointness over all folds...", flush=True)
    with ProcessPoolExecutor(max_workers=10) as ex:
        for k, res in ex.map(check_fold, [(k, id2tm) for k in range(10)]):
            report["fold_%d" % k] = res
            print("fold %d: %s" % (k, "PASS" if res["PASS"] else
                                   "FAIL " + json.dumps(res)), flush=True)

    ok = all(v["PASS"] for v in report.values())
    report["ALL_PASS"] = ok
    with open(os.path.join(ROOT, "work", "doublecheck_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("DOUBLECHECK:", "ALL PASS" if ok else "FAILURES FOUND", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
