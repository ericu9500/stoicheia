"""Restoration eval on the DeepMind `predictingthepast` "recent" comparison sets.

WHY THIS EXISTS: that paper's *recent* sets (`inscr_text_recent.jsonl`,
`pap_text_ups.jsonl`) are documents edited AFTER the compared systems' training
data was collected, so neither an 8B instruction-tuned Llama nor Ithaca nor this
model can have memorized them. They are therefore the only sets in that release
on which a three-way comparison is not confounded by contamination -- which is
the whole point of this paper's memorization argument.

SCORING IS THEIRS, NOT OURS. Reproduced verbatim from eval/scripts/eval_llama_text.py
in their release so all systems are scored identically:
  * CER = 1 - difflib.SequenceMatcher(None, ref, hyp).ratio()  -- a similarity
    ratio, NOT the Levenshtein-based CER used elsewhere in this repo. Numbers from
    this script are therefore NOT comparable to our clean/whole/strict CERs.
  * normalize: fold final sigma, strip spaces, '.', '·' and the digit '0'.
  * the hypothesis is truncated to the reference's length before scoring.
  * top-1 = normalized(truncate(beam_0)) == normalized(gold);
    top-20 = any of the 20 returned beams matches under the same rule.

Their input is lowercase, unaccented Greek with word spaces and exactly one
'[N letters missing]' placeholder. N counts LETTERS ONLY (spaces/punctuation are
excluded from the count), which matches our letters-only channel convention: word
division lives in the boundary plane, so a gap of N letters is N char positions.

  python insc/eval/restore_dsh.py --ckpt <ckpt> --data inscr_text_recent.jsonl \
      --out recent_inscr.json [--n 500] [--max-ctx 1024]
"""
from __future__ import annotations

import argparse, difflib, json, re, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from data.normalize import ALPHABET
from eval.intrinsic import load_model
from meta_vocab import UNK_REGION, UNK_CENTURY
from insc.eval.restore import beam_restore, MASK, UNK_BND

ALIST = list(ALPHABET)
AIDX = {c: i for i, c in enumerate(ALIST)}
AIDX.setdefault("ς", AIDX.get("σ"))
GAP_RE = re.compile(r"\[(\d+) letters? missing\]")

# ---------------------------------------------------------------- their metrics
def normalize_content(s: str) -> str:
    return re.sub(r"[ \.·0]", "", s.replace("ς", "σ"))

def truncate_to_real_length(pred: str, ref: str) -> str:
    return pred[:len(ref)]

def calculate_cer(reference: str, hypothesis: str) -> float:
    reference = normalize_content(reference)
    hypothesis = truncate_to_real_length(normalize_content(hypothesis), reference)
    return 1 - difflib.SequenceMatcher(None, reference, hypothesis).ratio()

# ---------------------------------------------------------------- encoding
def encode(text: str):
    """Greek text (letters + spaces) -> (char_ids, boundary_ids). Characters we
    cannot map are dropped; a space sets boundary=1 on the PRECEDING letter, which
    is how word division is represented in this model (never as a char token)."""
    chars, bnd = [], []
    for ch in text:
        if ch == " ":
            if bnd:
                bnd[-1] = 1
            continue
        i = AIDX.get(ch)
        if i is None:
            continue
        chars.append(i); bnd.append(0)
    return chars, bnd

def build(user_text: str, L: int, max_ctx: int):
    """-> (window chars, gap positions, boundary row) or None. Context is centered on
    the gap and truncated symmetrically to max_ctx so long documents stay in budget."""
    m = GAP_RE.search(user_text)
    if not m:
        return None
    pre, post = user_text[:m.start()], user_text[m.end():]
    cpre, bpre = encode(pre)
    cpost, bpost = encode(post)
    half = max(0, (max_ctx - L) // 2)
    if len(cpre) > half:
        cpre, bpre = cpre[-half:], bpre[-half:]
    if len(cpost) > half:
        cpost, bpost = cpost[:half], bpost[:half]
    gap = list(range(len(cpre), len(cpre) + L))
    window = np.array(cpre + [MASK] * L + cpost, dtype=np.int64)
    brow = np.array(bpre + [UNK_BND] * L + bpost, dtype=np.int64)
    # a space immediately before the gap is unknowable evidence about the gap's
    # first letter only in their format; keep it as given, consistent with restore.py
    return window, gap, brow

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all records")
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--max-ctx", type=int, default=1024)
    ap.add_argument("--shard", default="", help="i,k")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import os
    model, _ = load_model(os.path.expandvars(a.ckpt), device)   # returns (model, cfg)
    model.eval()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.shard:
        i, k = (int(x) for x in a.shard.split(","))
        rows = rows[i::k]
    if a.n:
        rows = rows[:a.n]

    per_len, skipped = {}, 0
    t1 = t20 = tot = 0
    for ri, r in enumerate(rows):
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        user, gold = msgs.get("user", ""), msgs.get("assistant", "")
        m = GAP_RE.search(user)
        if not m or not gold:
            skipped += 1; continue
        L = int(m.group(1))
        if not 1 <= L <= 10:
            skipped += 1; continue
        built = build(user, L, a.max_ctx)
        if built is None:
            skipped += 1; continue
        window, gap, brow = built
        cand = beam_restore(model, window, gap, brow, device, a.beam,
                            region_id=UNK_REGION, century_id=UNK_CENTURY)
        if not cand:
            skipped += 1; continue
        preds = [c[0] for c in cand]
        cer = calculate_cer(gold, preds[0])
        gn = normalize_content(gold)
        hit1 = int(normalize_content(truncate_to_real_length(preds[0], gold)) == gn)
        hit20 = int(any(normalize_content(truncate_to_real_length(p, gold)) == gn for p in preds[:20]))
        d = per_len.setdefault(L, dict(cer=[], t1=0, t20=0, n=0))
        d["cer"].append(cer); d["t1"] += hit1; d["t20"] += hit20; d["n"] += 1
        t1 += hit1; t20 += hit20; tot += 1
        if tot % 200 == 0:
            print(f"  {tot}/{len(rows)} CER={np.mean([c for v in per_len.values() for c in v['cer']]):.4f} "
                  f"top1={t1/tot:.4f}", flush=True)

    by_len = {L: dict(n=v["n"], CER=round(float(np.mean(v["cer"])), 4),
                      top1=round(v["t1"] / max(v["n"], 1), 4),
                      top20=round(v["t20"] / max(v["n"], 1), 4))
              for L, v in sorted(per_len.items())}
    # Their headline "Overall Average CER" is the mean over per-length means (their
    # summary prints one CER per length then averages), so report both that and the
    # sample-weighted mean rather than silently picking whichever looks better.
    macro = round(float(np.mean([v["CER"] for v in by_len.values()])), 4) if by_len else None
    micro = round(float(np.mean([c for v in per_len.values() for c in v["cer"]])), 4) if per_len else None
    res = dict(data=a.data, ckpt=a.ckpt, n=tot, skipped=skipped, beam=a.beam,
               CER_macro_over_lengths=macro, CER_micro=micro,
               top1=round(t1 / max(tot, 1), 4), top20=round(t20 / max(tot, 1), 4),
               by_length=by_len)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_length"}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
