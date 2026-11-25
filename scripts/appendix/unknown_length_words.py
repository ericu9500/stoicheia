"""E7b: unknown-length restoration for WHOLE-WORD lacunae — the distribution elastic masking is trained on
(train/noising.py::_elastic_rebuild masks whole words and pads AFTER the word end with ∅).

From the fold-3 test inscriptions (same documents as the frozen strict file), sample gaps of 1–3 consecutive fully
attested words. The model gets M letter slots (boundary unknown inside the gap; the word break before the gap and
the break at its end are given, as for a lacuna between visible word dividers) and fills letters or ∅ by greedy
most-confident-first mask-predict; word division inside the gap is read from the boundary head.
Conditions: known (M = k, ∅ disallowed) | elastic (M = k + 2 + floor(0.3k), the training distribution) |
plus5 (M = k + 5) | bound (M = 1.5k + 3 rounded up: only a loose physical upper bound).
Metrics: CER over the gap string (letters + internal spaces), exact match, |predicted letters − k|.

  python scripts/appendix/unknown_length_words.py --ckpt data/hub_pt/restoration-test3.pt \
      --samples insc/eval/frozen/strict_test_fold3_samples.json --n 1000 --out results/e7/unknown_length_words.json
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.intrinsic import load_model
from insc.eval.restore_strict import ALIST, MASK, UNK_BND, canon, levenshtein, text_to_planes
from scripts.appendix.unknown_length import forward
BLANK = 25


def make_samples(frozen, n, seed=0):
    docs = {}
    for s in json.load(open(frozen)):
        docs.setdefault(s["phi_id"], s["text"])
    rng = np.random.default_rng(seed); out = []
    keys = sorted(docs)
    while len(out) < n:
        t = docs[keys[int(rng.integers(len(keys)))]]
        words = t.split(" ")
        nw = int(rng.integers(1, 4))
        if len(words) < nw + 4:
            continue
        i = int(rng.integers(2, len(words) - nw - 1))
        gap = words[i:i + nw]
        if any((not w) or any(c not in "αβγδεζηθικλμνξοπρστυφχψωσ" for c in w) for w in gap):
            continue
        if any("-" in w for w in words[i - 1:i + nw + 1]):
            continue
        pre = " ".join(words[:i]) + " "; post = " " + " ".join(words[i + nw:])
        out.append(dict(pre=pre, post=post, gold=" ".join(gap)))
    return out


def decode(model, pre, post, M, allow_blank, device):
    full = pre + "α" * M + post
    ids, bnd, pos = text_to_planes(full)
    npre = sum(1 for q in pos if q < len(pre))
    g = list(range(npre, npre + M))
    ids = ids.copy(); ids[g] = MASK
    bnd = np.where(bnd == UNK_BND, UNK_BND, np.minimum(bnd, 2)); bnd[g] = UNK_BND
    vocab = list(range(24)) + ([BLANK] if allow_blank else [])
    rem = set(g)
    while rem:
        lp, _ = forward(model, ids, bnd, device)
        order = sorted(rem); sub = lp[order][:, vocab]; best = sub.max(-1)
        r = int(best.values.argmax()); ids[order[r]] = vocab[int(best.indices[r])]; rem.discard(order[r])
    keep = np.array([not (i in set(g) and ids[i] == BLANK) for i in range(len(ids))])
    letters = [i for i in g if ids[i] != BLANK]
    if not letters:
        return ""
    ids2, bnd2 = ids[keep], bnd[keep]
    new_g = [int(keep[:i].sum()) for i in letters]
    _, pb = forward(model, ids2, bnd2, device)
    p_end = (pb[:, 1] + pb[:, 2]).cpu().numpy()
    return "".join(ALIST[ids2[q]] + (" " if j < len(new_g) - 1 and p_end[q] > 0.5 else "") for j, q in enumerate(new_g))


def decode_joint(model, pre, post, M, device):
    """Length-marginal decoding: for every candidate letter count k' = 1..M, greedily fill the first k' slots
    (surplus slots stay masked), then score the complete hypothesis over ALL M slots in one final forward:
    sum log P(letter) on the k' slots + sum log P(∅) on the M-k' surplus slots. Every hypothesis is scored on the
    same M slots, so totals are comparable across lengths. Returns (best string, best k', rank of each k')."""
    full = pre + "α" * M + post
    ids0, bnd0, pos = text_to_planes(full)
    npre = sum(1 for q in pos if q < len(pre))
    g = list(range(npre, npre + M))
    ids0 = ids0.copy(); ids0[g] = MASK
    bnd0 = np.where(bnd0 == UNK_BND, UNK_BND, np.minimum(bnd0, 2)); bnd0[g] = UNK_BND
    best = None; scores = {}
    for kk in range(1, M + 1):
        ids = ids0.copy(); rem = set(g[:kk])
        while rem:
            lp, _ = forward(model, ids, bnd0, device)
            order = sorted(rem); sub = lp[order][:, :24]; bv = sub.max(-1)
            r = int(bv.values.argmax()); ids[order[r]] = int(bv.indices[r]); rem.discard(order[r])
        lp, pb = forward(model, ids, bnd0, device)
        tot = float(sum(lp[q, ids[q]] for q in g[:kk]) + sum(lp[q, BLANK] for q in g[kk:]))
        scores[kk] = tot
        if best is None or tot > best[0]:
            p_end = (pb[:, 1] + pb[:, 2]).cpu().numpy()
            s = "".join(ALIST[ids[q]] + (" " if j < kk - 1 and p_end[q] > 0.5 else "") for j, q in enumerate(g[:kk]))
            best = (tot, s, kk)
    return best[1], best[2], scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--samples", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out", required=True); ap.add_argument("--shard")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.expandvars(a.ckpt), device); model.cfg.attn_impl = "sdpa"; model.eval()
    samples = make_samples(a.samples, a.n)
    if a.shard:
        i, k = map(int, a.shard.split(",")); samples = samples[i::k]
    conds = {"known": lambda k: (k, False), "elastic": lambda k: (k + 2 + int(0.3 * k), True),
             "plus5": lambda k: (k + 5, True), "bound": lambda k: (math.ceil(1.5 * k) + 3, True)}
    res = {c: [] for c in conds}
    for si, s in enumerate(samples):
        gold = canon(s["gold"]); k = len(gold.replace(" ", ""))
        for c, f in conds.items():
            M, bl = f(k)
            pred = decode(model, s["pre"], s["post"], M, bl, device)
            res[c].append(dict(gold=gold, pred=pred, k=k, nwords=gold.count(" ") + 1,
                               cer=levenshtein(pred, gold) / len(gold), exact=pred == gold,
                               len_err=abs(len(pred.replace(" ", "")) - k)))
            if bl:
                pj, kj, sc = decode_joint(model, s["pre"], s["post"], M, device)
                rank = 1 + sum(v > sc[k] for v in sc.values())
                res.setdefault(c + "_joint", []).append(dict(gold=gold, pred=pj, k=k, nwords=gold.count(" ") + 1,
                               cer=levenshtein(pj, gold) / len(gold), exact=pj == gold,
                               len_err=abs(kj - k), true_len_rank=rank))
        if (si + 1) % 50 == 0:
            print(si + 1, {c: round(float(np.mean([r["cer"] for r in v])), 4) for c, v in res.items()}, flush=True)
    summ = {c: dict(n=len(v), CER=round(100 * float(np.mean([r["cer"] for r in v])), 2),
                    exact=round(100 * float(np.mean([r["exact"] for r in v])), 2),
                    mean_abs_len_err=round(float(np.mean([r["len_err"] for r in v])), 3),
                    exact_len_pct=round(100 * float(np.mean([r["len_err"] == 0 for r in v])), 1),
                    true_len_top3_pct=(round(100 * float(np.mean([r["true_len_rank"] <= 3 for r in v])), 1) if "true_len_rank" in v[0] else None),
                    by_nwords={w: round(100 * float(np.mean([r["cer"] for r in v if r["nwords"] == w])), 2) for w in (1, 2, 3)})
            for c, v in res.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, samples=res), ensure_ascii=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
