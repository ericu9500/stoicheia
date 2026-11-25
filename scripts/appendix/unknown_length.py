"""E7: restoration when the lacuna's extent is NOT known (paper Limitations: "unknown-length gaps
are not evaluated here, although the elastic masking trains the capability").

Same frozen strict samples (Ithaca char space). The gap holds k letters (gold, spaces excluded). The
model receives M mask slots with the boundary channel unknown inside the gap and must fill letters or
the empty symbol ∅ (blank_id 25); word division is read from the boundary head on the filled
sequence. Conditions, all decoded identically (greedy most-confident-first mask-predict):
  known       M = k                    (length given; ∅ disallowed)
  elastic     M = k + 2 + floor(0.3k)  (the over-provisioning distribution seen in training)
  plus5       M = k + 5
  unknown10   M = 10 + 2 + 3 = 15      (only an upper bound of 10 letters is known)
Metrics: CER over the gap string (spaces count, as in the strict protocol), exact match, and
|predicted letters - k|.

  python scripts/appendix/unknown_length.py --ckpt data/hub_pt/restoration-test3.pt \
      --samples insc/eval/frozen/strict_test_fold3_samples.json --out results/e7/unknown_length.json
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.intrinsic import load_model
from insc.eval.restore_strict import (ALIST, A_IDX, MASK, UNK_BND, UNK_CENTURY, UNK_DIA, UNK_PUNCT,
                                      UNK_REGION, canon, levenshtein, text_to_planes)

BLANK = 25


@torch.no_grad()
def forward(model, ids, bnd, device):
    T = len(ids)
    b = dict(input_ids=torch.tensor(ids, device=device)[None], boundary=torch.tensor(bnd, device=device)[None],
             dia=torch.full((1, T), UNK_DIA, device=device), punct=torch.full((1, T), UNK_PUNCT, device=device),
             region=torch.full((1, T), UNK_REGION, device=device), century=torch.full((1, T), UNK_CENTURY, device=device),
             seg_id=torch.ones(1, T, dtype=torch.long, device=device))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        o = model(b)
    return torch.log_softmax(o["char"][0].float(), -1), torch.softmax(o["boundary"][0].float(), -1)


def decode(model, pre, post, M, allow_blank, device):
    full = pre + "α" * M + post
    ids, bnd, pos = text_to_planes(full)
    s = len(pre)
    npre = sum(1 for q in pos if q < s)
    gidx = list(range(npre, npre + M))
    ids = ids.copy(); ids[gidx] = MASK
    bnd = np.where(bnd == UNK_BND, UNK_BND, np.minimum(bnd, 2)); bnd[gidx] = UNK_BND
    lead_free = len(pre) > 0 and not pre.endswith(" ") and npre > 0
    if lead_free:
        bnd[npre - 1] = UNK_BND
    vocab = list(range(24)) + ([BLANK] if allow_blank else [])
    rem = set(gidx)
    while rem:
        lp, _ = forward(model, ids, bnd, device)
        sub = lp[sorted(rem)][:, vocab]
        best = sub.max(-1)
        r = int(best.values.argmax())
        p = sorted(rem)[r]
        ids[p] = vocab[int(best.indices[r])]
        rem.discard(p)
    letters = [i for i in gidx if ids[i] != BLANK]
    # remove blanks from the sequence, then read word division
    keep = np.array([i not in set(gidx) or ids[i] != BLANK for i in range(len(ids))])
    ids2, bnd2 = ids[keep], bnd[keep]
    new_g = [int(np.sum(keep[:i])) for i in letters]
    if not new_g:
        return ""
    _, pb = forward(model, ids2, bnd2, device)
    p_end = (pb[:, 1] + pb[:, 2]).cpu().numpy()
    out = ""
    if lead_free and p_end[new_g[0] - 1] > 0.5:
        out += " "
    for j, g in enumerate(new_g):
        out += ALIST[ids2[g]]
        last = j == len(new_g) - 1
        if (not last and p_end[g] > 0.5) or (last and post and not post.startswith(" ") and p_end[g] > 0.5):
            out += " "
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.expandvars(a.ckpt), device)
    model.cfg.attn_impl = "sdpa"; model.eval()
    samples = json.loads(Path(a.samples).read_text())
    if a.shard:
        i, k = (int(x) for x in a.shard.split(",")); samples = samples[i::k]
    conds = {"known": lambda k: (k, False), "elastic": lambda k: (k + 2 + int(0.3 * k), True),
             "plus5": lambda k: (k + 5, True), "unknown10": lambda k: (15, True)}
    res = {c: [] for c in conds}
    for si, smp in enumerate(samples):
        t, s, L = smp["text"], smp["start"], smp["L"]
        gold = canon(smp["gold"]); pre, post = t[:s], t[s + L:]
        # the gap string's letters are placed as a contiguous letter block; surrounding spaces
        # belong to the free leading/trailing division sites (as in the strict evaluator)
        k = len(gold.replace(" ", ""))
        if k == 0:
            continue
        for c, f in conds.items():
            M, blank = f(k)
            pred = decode(model, pre, post, M, blank, device)
            res[c].append(dict(L=L, k=k, gold=gold, pred=pred, cer=levenshtein(pred, gold) / len(gold),
                               exact=pred == gold, len_err=abs(len(pred.replace(" ", "")) - k)))
        if (si + 1) % 50 == 0:
            print(si + 1, {c: round(float(np.mean([r["cer"] for r in v])), 4) for c, v in res.items()}, flush=True)
    summ = {c: dict(n=len(v), CER=round(100 * float(np.mean([r["cer"] for r in v])), 2),
                    exact=round(100 * float(np.mean([r["exact"] for r in v])), 2),
                    mean_abs_len_err=round(float(np.mean([r["len_err"] for r in v])), 3),
                    by_L={L: round(100 * float(np.mean([r["cer"] for r in v if r["L"] == L])), 2)
                          for L in sorted({r["L"] for r in v})}) for c, v in res.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, samples=res), ensure_ascii=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
