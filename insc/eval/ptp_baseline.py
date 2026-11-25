"""Aeneas-architecture Greek model (DeepMind `predictingthepast`, Nature 2025) on
our two frozen comparison sets.

Two modes, matching the two tables the paper compares systems in:

  --samples <strict_test_fold3_samples.json>   STRICT protocol: the same frozen
      3,000-sample file every other system reads. Gap = '?'*L at [start,start+L),
      spaces count as characters, predictions keep spaces, sigma/koppa folded,
      Levenshtein CER + exact top-1/top-20. Mirrors ithaca_baseline.run_strict.

  --dsh <inscr_text_recent.jsonl>              RECENT (uncontaminated) set: the
      DSH-41(1)-comparison release format ('[N letters missing]', N counts letters only).
      Scored with the verified port of their metric (difflib similarity ratio,
      normalize = fold final sigma + strip ' .·0', truncate hypothesis).

Restoration only; the model's retrieval/attribution capabilities are unused.
beam_width=20 in both modes to match every other system in both tables.

The model has a fixed context of 768 characters (inference.TEXT_LEN); longer
inputs are cropped symmetrically around the gap, mirroring the old harness.

  .venv-ptp/bin/python insc_eval/ptp_baseline.py --ckpt models/ithaca_153143996_2.pkl \
      --samples ../strict_test_fold3_samples.json --out out.json --shard 0,40
"""
from __future__ import annotations

import argparse, difflib, json, os, re, time
from pathlib import Path

import numpy as np

STRICT_FOLD = {"ς": "σ", "ϲ": "σ", "ϙ": "κ", "ϛ": "σ"}
GAP_RE = re.compile(r"\[(\d+) letters? missing\]")
CTX = 750  # < inference.TEXT_LEN with margin for SOS/padding


def canon(s):
    return "".join(STRICT_FOLD.get(c, c) for c in s)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---- their metric, verified port (reproduces released per-record CERs exactly)
def dsh_norm(s):
    return re.sub(r"[ \.·0]", "", s.replace("ς", "σ"))


def dsh_cer(ref, hyp):
    ref = dsh_norm(ref)
    hyp = dsh_norm(hyp)[: len(ref)]
    return 1 - difflib.SequenceMatcher(None, ref, hyp).ratio()


# ---------------------------------------------------------------- model
G = {}


def init(ckpt):
    # Mirrors inference_example.load_checkpoint(path, 'greek') exactly.
    import pickle

    import jax
    from predictingthepast.eval import inference
    from predictingthepast.models.model import Model
    from predictingthepast.util import alphabet as util_alphabet

    with open(ckpt, "rb") as f:
        checkpoint = pickle.load(f)
    params = jax.device_put(checkpoint["params"])
    model = Model(**checkpoint["model_config"])
    G.update(inference=inference, forward=model.apply, params=params,
             alphabet=util_alphabet.GreekAlphabet(),
             vocab=checkpoint["model_config"]["vocab_char_size"])


def crop(text, s, L):
    """Center a window of <=CTX chars on the gap [s, s+L)."""
    if len(text) <= CTX:
        return text, s
    half = (CTX - L) // 2
    lo = max(0, s - half)
    hi = min(len(text), lo + CTX)
    lo = max(0, hi - CTX)
    return text[lo:hi], s - lo


def restore(text, beam, scores_out=None):
    r = G["inference"].restore(
        text, forward=G["forward"], params=G["params"], alphabet=G["alphabet"],
        vocab_char_size=G["vocab"], beam_width=beam)
    hyps = []
    for p in r.predictions[:beam]:
        idx = p.restored if p.restored else r.missing
        hyps.append("".join(p.text[i] for i in idx if i < len(p.text)))
        if scores_out is not None:   # per-hypothesis score (exp beam logprob)
            scores_out.append(float(p.score))
    if not hyps and r.top_prediction:
        hyps = ["".join(r.top_prediction[i] for i in r.missing
                        if i < len(r.top_prediction))]
        if scores_out is not None:
            scores_out.append(None)
    return hyps


# ---------------------------------------------------------------- modes
def run_strict(a):
    samples = json.loads(Path(os.path.expandvars(a.samples)).read_text())
    if a.lengths:
        want = {int(x) for x in a.lengths.split(",")}
        samples = [s for s in samples if s["L"] in want]
    if a.shard:
        i, k = (int(x) for x in a.shard.split(","))
        samples = samples[i::k]
    print(f"{len(samples)} strict samples", flush=True)
    rows, t0, n_err = {}, time.time(), 0
    per_sample = []   # per-sample top-20 + scores (error analysis, calibration)
    for i, s in enumerate(samples):
        text = s["text"][:s["start"]] + "?" * s["L"] + s["text"][s["start"] + s["L"]:]
        text, _ = crop(text, s["start"], s["L"])
        gold = canon(s["gold"])
        sc = []
        try:
            hyps = [canon(h) for h in restore(text, a.beam, sc)]
        except Exception as e:
            n_err += 1
            hyps = []
            sc = []
            print(f"  ERR at {i}: {str(e)[:80]}", flush=True)
        per_sample.append(dict(phi_id=s["phi_id"], L=s["L"], start=s["start"], gold=gold,
                               preds=hyps, scores=sc))
        r = rows.setdefault(s["L"], dict(n=0, cers=[], t1=0, t20=0))
        if hyps:
            r["cers"].append(levenshtein(hyps[0], gold) / max(len(gold), 1))
            r["t1"] += int(hyps[0] == gold)
            r["t20"] += int(any(h == gold for h in hyps))
            r["n"] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)} ({(time.time()-t0)/(i+1):.1f}s/sample)",
                  flush=True)
    out = dict(model="aeneas_greek_2025", protocol="strict", errors=n_err,
               per_L={L: dict(n=r["n"], CER=round(float(np.mean(r["cers"])), 4),
                              top1=round(r["t1"] / max(r["n"], 1), 4),
                              top20=round(r["t20"] / max(r["n"], 1), 4))
                      for L, r in sorted(rows.items())})
    out["samples"] = per_sample
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


def run_dsh(a):
    rows_in = [json.loads(l) for l in open(os.path.expandvars(a.dsh))]
    if a.shard:
        i, k = (int(x) for x in a.shard.split(","))
        rows_in = rows_in[i::k]
    print(f"{len(rows_in)} recent-set samples", flush=True)
    per, t0, n_err = {}, time.time(), 0
    for i, rec in enumerate(rows_in):
        msgs = {m["role"]: m["content"] for m in rec["messages"]}
        user, gold = msgs.get("user", ""), msgs.get("assistant", "")
        m = GAP_RE.search(user)
        if not m or not gold:
            continue
        L = int(m.group(1))
        if not 1 <= L <= 10:
            continue
        # Gap construction copied from their own eval_ithaca_text.py: the number
        # of '?' slots is len(gold) -- the FULL gold INCLUDING SPACES -- not the
        # N letters of the placeholder. 58% of golds contain internal spaces
        # (388/398 at L=10); allocating only N slots leaves an Ithaca-style
        # model no room to emit both the word divisions and the letters, and
        # collapsed long-gap accuracy to ~1% in the first version of this
        # harness. Their released Ithaca predictions used this construction,
        # so mirroring it is also what makes the comparison symmetric.
        slots = len(gold)
        text = user[:m.start()] + "?" * slots + user[m.end():]
        text, _ = crop(text, m.start(), slots)
        try:
            hyps = restore(text, a.beam)
        except Exception as e:
            n_err += 1
            print(f"  ERR at {i}: {str(e)[:80]}", flush=True)
            continue
        gn = dsh_norm(gold)
        pred = hyps[0] if hyps else ""
        r = per.setdefault(L, dict(n=0, cers=[], t1=0, t20=0))
        r["cers"].append(dsh_cer(gold, pred))
        r["t1"] += int(dsh_norm(pred)[: len(gn)] == gn)
        r["t20"] += int(any(dsh_norm(h)[: len(gn)] == gn for h in hyps[:20]))
        r["n"] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows_in)} ({(time.time()-t0)/(i+1):.1f}s/sample)",
                  flush=True)
    out = dict(model="aeneas_greek_2025", protocol="dsh_recent",
               scoring="DSH2026 difflib-ratio", errors=n_err,
               per_L={L: dict(n=r["n"], CER=round(float(np.mean(r["cers"])), 4),
                              top1=round(r["t1"] / max(r["n"], 1), 4),
                              top20=round(r["t20"] / max(r["n"], 1), 4))
                      for L, r in sorted(per.items())})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", default=None, help="strict frozen samples json")
    ap.add_argument("--dsh", default=None, help="recent-set jsonl (DSH 2026)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--lengths", default="")
    ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    init(os.path.expandvars(a.ckpt))
    if a.samples:
        run_strict(a)
    elif a.dsh:
        run_dsh(a)
    else:
        raise SystemExit("need --samples or --dsh")


if __name__ == "__main__":
    main()
