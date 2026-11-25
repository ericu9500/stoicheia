"""E5 + E6 for macronization on Norma: error analysis, calibration, selective prediction.

Inputs: meter.predict --norma outputs (<preds>.jsonl and <preds>_probs.json) for one model.
  python analysis/macron_analysis.py --preds results/e0_meter/macronizer_norma_preds.jsonl \
      --name macronizer_released --out results/analysis/macron_released.json

Reports (dev and test)
  accuracy / balanced accuracy overall and by vowel (α ι υ), gold class, source work,
  accent on the vowel, and word-final vs non-final syllable
  calibration of P(long): ECE, Brier, reliability table
  selective prediction: accuracy when abstaining on the least confident positions at coverage
  100..10 %, incl. 60 % (the rule-based Norma macronizer's coverage, paper §7)
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from data.normalize import ALPHABET, unpack_dia
from meter.dataset import encode_plain
from meter.marks import parse_macron_line
from meter.norma_data import load_norma

LONG = 0   # meter.norma_score convention: class 0 = long
VOWELS = set("αεηιουω")


def bal(y, p):
    y, p = np.asarray(y), np.asarray(p)
    r = [float((p[y == c] == c).mean()) for c in (0, 1) if (y == c).any()]
    return round(100 * float(np.mean(r)), 2) if r else None


def summarize(rows):
    if not rows:
        return dict(n=0)
    y = [r["gold"] for r in rows]; p = [r["pred"] for r in rows]
    return dict(n=len(rows), acc=round(100 * float(np.mean(np.equal(y, p))), 2), bal_acc=bal(y, p))


def word_final_syllable(rec, k):
    """True if no vowel letter follows letter k before the end of its word."""
    j = k
    while True:
        if rec.boundary[j] >= 1 or j == len(rec) - 1:
            return True
        j += 1
        if ALPHABET[rec.chars[j]] in VOWELS:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--norma-source", default="hf")
    a = ap.parse_args()
    norma = load_norma(a.norma_source)
    preds = [json.loads(l) for l in open(a.preds, encoding="utf-8")]
    probs = json.load(open(a.preds.replace(".jsonl", "") + "_probs.json"))
    rep = dict(name=a.name)
    for split in ("dev", "test"):
        items = [d for d in norma[split] if d["task"] == "macronize"]
        sp = [p for p in preds if p["split"] == split]
        prob_it = iter([p for p in probs if p["split"] == split])
        assert len(sp) == len(items), (split, len(sp), len(items))
        rows = []
        for d, p in zip(items, sp):
            plain, _ = parse_macron_line(d["text"])
            rec = encode_plain(plain)
            pl = next(prob_it) if rec is not None else None
            for j, ((k, _, gv), (_, _, pv)) in enumerate(zip(p["gold"], p["pred"])):
                k = int(k)
                if rec is not None:
                    vowel = ALPHABET[rec.chars[k]]
                    acc_ = unpack_dia(int(rec.dia[k]))[0]
                    final = word_final_syllable(rec, k)
                else:
                    vowel, acc_, final = "?", 0, None
                rows.append(dict(gold=int(gv), pred=int(pv), vowel=vowel, source=d["source"],
                                 accent=["none", "acute", "grave", "circumflex"][acc_], final_syll=final,
                                 p_long=float(pl["seq"][j][1]) if pl is not None else None))
        s = dict(overall=summarize(rows))
        for key in ("vowel", "source", "accent", "final_syll"):
            g = defaultdict(list)
            for r in rows:
                g[str(r[key])].append(r)
            s[f"by_{key}"] = {k: summarize(v) for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}
        s["by_gold_class"] = {"long": summarize([r for r in rows if r["gold"] == LONG]),
                              "short": summarize([r for r in rows if r["gold"] != LONG])}
        cal = [r for r in rows if r["p_long"] is not None]
        if cal:
            pl = np.array([r["p_long"] for r in cal]); yl = np.array([r["gold"] == LONG for r in cal])
            pred_long = np.array([r["pred"] == LONG for r in cal])
            conf = np.where(pred_long, pl, 1 - pl); correct = (pred_long == yl).astype(float)
            ece, table = 0.0, []
            edges = np.linspace(0.5, 1.0, 11)
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
                if m.any():
                    ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
                    table.append(dict(bin=f"{lo:.2f}-{hi:.2f}", n=int(m.sum()), conf=round(float(conf[m].mean()), 3),
                                      acc=round(float(correct[m].mean()), 3)))
            order = np.argsort(-conf); sel = []
            for c in (100, 90, 80, 70, 60, 50, 40, 30, 20, 10):
                idx = order[:max(1, int(round(len(cal) * c / 100)))]
                sel.append(dict(coverage=c, acc=round(100 * float(correct[idx].mean()), 2),
                                bal_acc=bal([cal[i]["gold"] for i in idx], [cal[i]["pred"] for i in idx])))
            s["calibration"] = dict(n=len(cal), ece=round(float(ece), 4),
                                    brier=round(float(np.mean((pl - yl) ** 2)), 4), reliability=table, selective=sel)
        rep[split] = s
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({sp: rep[sp]["overall"] for sp in ("dev", "test")}))


if __name__ == "__main__":
    main()
