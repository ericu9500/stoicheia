"""STRICT protocol restoration eval for PAPYRI -- same methodology as restore_strict.py
(inscriptions), applied to whole DDbDP documents instead of pre-split segments.

Protocol (identical to restore_strict.py, no relaxations):
  - unit: L consecutive characters (letters and/or spaces) of the document's flattened
    text -- SPACES COUNT, exactly as for inscriptions. 8 letters + 2 word boundaries
    inside the gap = L10, not L8.
  - context: the WHOLE papyrus document (up to 3000 flattened chars; ctx=3200 letters
    in eval_ours guarantees every sample's full document fits in the model's window --
    no truncation to a local span around the gap).
  - our model predicts word division inside the gap (boundary head); division errors
    count in CER and break exact-match, same as inscriptions.
  - documents containing '…' (extent="unknown" gaps -- indeterminate length) are
    EXCLUDED entirely: their surrounding lacunae have no fixed width, which would make
    the "other lacunae are known-width unknowns" assumption invalid. Only '-' runs
    (known-width lost text) are kept as in-context lacunae, exactly like inscriptions.
  - unlike inscriptions' ithaca_text (already flattened: lowercase, accentless, only
    letters/space/'-'), papyri_clean.jsonl keeps real accents/case/extra punctuation
    (see clean_papyri.py). flatten() folds to the SAME reduced alphabet (24 base
    letters + space + '-') that restore_strict.py's text_to_planes()/eval_ours()
    already operate on, so all of its beam-search + scoring machinery is reused
    UNCHANGED -- this file only supplies papyri-specific data loading.

Modes (same CLI shape as restore_strict.py):
  --make-samples: python insc_eval/restore_strict_papyri.py --make-samples --split test \
      --n 300 --samples $INS_DATA/strict_test_samples_papyri.json
  --ckpt:         python insc_eval/restore_strict_papyri.py --ckpt <best.pt> \
      --samples $INS_DATA/strict_test_samples_papyri.json --out <out.json>
"""
from __future__ import annotations

import argparse, json, os, re, sys, unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(2, str(Path(__file__).resolve().parent))
from data.normalize import ALPHABET
from restore_strict import eval_ours, levenshtein, canon  # reused verbatim

A_IDX = {c: i for i, c in enumerate(ALPHABET)}
PAP_JSONL = os.path.expandvars("$AGD_DATA/data/papyri_clean.jsonl")

# codepoints that fold onto the 24-letter alphabet (mirrors data/normalize.py's
# _EXTRA_BASE, kept local so this file has no hidden coupling beyond ALPHABET itself)
_EXTRA = {"ς": "σ", "ϲ": "σ", "Ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϕ": "φ",
          "ϰ": "κ", "ϱ": "ρ", "ϖ": "π"}
_PUNCT_TO_SPACE = str.maketrans({c: " " for c in "·,."})


def to_base_greek(ch):
    """Lowercase + NFD-strip accents + fold sigma/symbol variants -> one of the 24
    base letters (chars-plane alphabet), or None if not a modeled Greek letter."""
    lo = ch.lower()
    lo = _EXTRA.get(lo, lo)
    if lo in A_IDX:
        return lo
    dec = unicodedata.normalize("NFD", lo)
    base = _EXTRA.get(dec[0], dec[0]) if dec else ""
    return base if base in A_IDX else None


def flatten(text):
    """Fold a papyri_clean.jsonl 'text' string down to restore_strict.py's alphabet:
    24 base letters + space + '-' (known-width lacuna) only -- same vocabulary as
    inscriptions' ithaca_text, so text_to_planes()/eval_ours() need no changes."""
    t = text.translate(_PUNCT_TO_SPACE)
    out = []
    for ch in t:
        if ch in (" ", "-"):
            out.append(ch)
        else:
            b = to_base_greek(ch)
            out.append(b if b is not None else " ")
    flat = re.sub(r" +", " ", "".join(out)).strip()
    return flat


def split_of(tm):
    s = str(tm).strip()
    if not s or not s[-1].isdigit():
        return "train"
    test_d = os.environ.get("INSC_TEST_DIGIT", "3")
    val_d = os.environ.get("INSC_VAL_DIGIT", "4")
    return {val_d: "val", test_d: "test"}.get(s[-1], "train")


def load_papyri(split, exclude=None):
    """Whole papyri documents, flattened. Docs with '…' (unknown-width gap)
    dropped entirely (see module docstring)."""
    bad_tm = set()
    if exclude:
        j = json.loads(Path(os.path.expandvars(exclude)).read_text())
        bad_tm = {str(x[0]) for x in j["contaminated"] if str(x[0]) != "POSITIVE_CONTROL"}
    out = []
    with open(PAP_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tm = str(r.get("TM", ""))
            sp = split_of(tm)
            if sp != split or tm in bad_tm or "…" in (r.get("text") or ""):
                continue
            t = flatten(r.get("text") or "")
            if 50 <= len(t) <= 3000:
                out.append(dict(phi_id=tm, text=t))
    return out


def make_samples(split, n, lengths, exclude, out_path):
    recs = load_papyri(split, exclude)
    rng = np.random.default_rng(1234)
    rng.shuffle(recs)
    print(f"{len(recs)} whole papyri documents in {split} (decontaminated, "
          f"flattened, 50-3000 chars, no '…')")
    samples = []
    for L in lengths:
        lrng = np.random.default_rng(0 + L)
        tot = 0
        for r in recs:
            t = r["text"]
            if len(t) <= L + 8:
                continue
            s = int(lrng.integers(4, len(t) - L - 4))
            gap = t[s:s + L]
            if ("-" in gap or "  " in gap or gap.strip(" ") == "" or
                    (s > 0 and t[s - 1] == " " and gap[0] == " ") or
                    (t[s + L:s + L + 1] == " " and gap[-1] == " ")):
                continue
            samples.append(dict(phi_id=r["phi_id"], L=L, start=s, gold=gap, text=t))
            tot += 1
            if tot >= n:
                break
        print(f"L={L}: {tot} samples")
    Path(os.path.expandvars(out_path)).write_text(json.dumps(samples))
    print(f"wrote {len(samples)} samples -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-samples", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lengths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=3200,
                    help="letter-window width around the gap; >= max doc length (3000 "
                         "chars, fewer letters after flattening) guarantees full-document context")
    ap.add_argument("--shard", default=None, help="i,k -> process samples[i::k]")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    if a.make_samples:
        make_samples(a.split, a.n, lengths, a.exclude, a.samples)
    elif a.ckpt:
        eval_ours(a.ckpt, a.samples, a.out, beam_width=a.beam, ctx=a.ctx,
                  lengths_filter=set(lengths), shard=a.shard)
    else:
        raise SystemExit("need --make-samples or --ckpt")


if __name__ == "__main__":
    main()
