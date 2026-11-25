"""Encode the macron-data corpora into letter-plane npz stores.

  python -m meter.encode [--out $METER_DATA/encoded] [--src $MACRONIZER_SRC] [--norma-source hf]

Sources (all under --src/data, except Norma -- see --norma-source):
  macron TSVs  plain \t marked   (verse silver + OGA prose silver)
  dev.txt      marked lines      (763 Aristophanic verses, macron dev set)
  scanner/corpus_v3.tsv           work \t meter \t bracketed verse

Anything whose letter stream overlaps the Norma benchmark (both tasks, dev+test) or
dev.txt is EXCLUDED from training stores: a record is dropped if any 20-letter
shingle of an eval line occurs in it (shorter eval lines: exact letter-stream match).
Slightly over-eager by design — dropping a few extra silver lines is free, leakage
is not.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

from meter.backbone import ALPHABET  # noqa: F401 (puts STOICHEIA_ROOT on sys.path)
from meter.dataset import encode_macron_line, encode_scan_line, save_records
from meter.marks import parse_macron_line, parse_scan_line
from meter.norma_data import add_norma_source_arg, load_norma

MACRON_TSVS = ["hypotactic", "drama_ia6", "drama_ia6_tet", "anthology",
               "nonnus_quintus", "babrius_chol", "theocritus_doric",
               "theocritus_other", "sweep1_hex", "sweep1_ia6", "sweep1_chol",
               "sweep1_eleg", "oga_0", "oga_1", "oga_2", "oga_3"]
SHINGLE = 20


def letters_of(rec) -> str:
    return "".join(ALPHABET[c] for c in rec.chars)


ALPHABET_SET = set(ALPHABET) | {"ς", "ϲ"}


def _letters_only(plain: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", plain).lower()
                   if ch in ALPHABET_SET)


def eval_letter_streams(src: Path, norma_source: str = "hf"):
    """Letter streams of every eval line (Norma macronize+syllabify -- dev+test both,
    so the exclusion screen covers everything Norma could ever score us against --
    plus dev.txt)."""
    streams = []
    norma = load_norma(norma_source)
    for d in norma["dev"] + norma["test"]:
        parsed = (parse_scan_line(d["text"]) if d["task"] == "syllabify"
                  else parse_macron_line(d["text"]))
        if parsed is not None:
            streams.append(_letters_only(parsed[0]))
    for line in open(src / "data/dev.txt", encoding="utf-8"):
        streams.append(_letters_only(parse_macron_line(line.rstrip("\n"))[0]))
    return [s for s in streams if s]


def build_screen(streams):
    shingles, exact = set(), set()
    for s in streams:
        s = s.replace("ς", "σ").replace("ϲ", "σ")
        if len(s) >= SHINGLE:
            for i in range(len(s) - SHINGLE + 1):
                shingles.add(s[i:i + SHINGLE])
        else:
            exact.add(s)
    return shingles, exact


def is_contaminated(letters: str, shingles, exact) -> bool:
    if letters in exact:
        return True
    for i in range(len(letters) - SHINGLE + 1):
        if letters[i:i + SHINGLE] in shingles:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expandvars(os.environ.get(
        "MACRONIZER_SRC", "$MACRONIZER_SRC")))
    ap.add_argument("--out", default=None)
    add_norma_source_arg(ap)
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out or os.path.join(os.environ["METER_DATA"], "encoded"))
    out.mkdir(parents=True, exist_ok=True)

    print("building eval exclusion screen ...", flush=True)
    shingles, exact = build_screen(eval_letter_streams(src, a.norma_source))
    print(f"  {len(shingles):,} shingles, {len(exact)} exact keys", flush=True)

    stats = {}

    def finish(name, kept, dropped, excluded, works=None):
        save_records(out / f"{name}.npz", kept, works)
        n_mac = sum(int((r.y_mac != -100).sum()) for r in kept)
        stats[name] = dict(records=len(kept), dropped=dropped, excluded=excluded,
                           letters=sum(len(r) for r in kept), mac_labels=n_mac)
        print(f"  {name}: kept={len(kept):,} dropped={dropped:,} "
              f"excluded={excluded:,} mac_labels={n_mac:,}", flush=True)

    # ---- macron TSVs (train)
    for name in MACRON_TSVS:
        path = src / "data" / f"{name}.tsv"
        if not path.exists():
            print(f"  {name}: MISSING, skipped", flush=True)
            continue
        kept, dropped, excluded = [], 0, 0
        for line in open(path, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                dropped += 1
                continue
            rec = encode_macron_line(parts[1])
            if rec is None:
                dropped += 1
                continue
            if is_contaminated(letters_of(rec), shingles, exact):
                excluded += 1
                continue
            kept.append(rec)
        finish(name, kept, dropped, excluded)

    # ---- macron dev (no exclusion screen — it IS an eval set)
    kept, dropped = [], 0
    for line in open(src / "data/dev.txt", encoding="utf-8"):
        rec = encode_macron_line(line.rstrip("\n"))
        if rec is None:
            dropped += 1
            continue
        kept.append(rec)
    finish("dev_aristophanes", kept, dropped, 0)

    # ---- scanner corpus (train/dev/test split by work happens at load time)
    kept, works, dropped, excluded = [], [], 0, 0
    for line in open(src / "data/scanner/corpus_v3.tsv", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3 or parts[0] == "?":
            dropped += 1
            continue
        rec = encode_scan_line(parts[2])
        if rec is None:
            dropped += 1
            continue
        if is_contaminated(letters_of(rec), shingles, exact):
            excluded += 1
            continue
        kept.append(rec)
        works.append(parts[0])
    finish("scan_corpus", kept, dropped, excluded, works)

    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("done:", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
