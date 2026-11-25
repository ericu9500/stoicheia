"""E5 + E6 for documentary restoration: error analysis, calibration and selective prediction.

Input: one or more per-sample strict-eval JSONs (insc/eval/restore_strict.py with per-sample dump, or ablate/tok_insc.py strict), possibly shards of one run.

  python analysis/restoration_analysis.py --name v2_fold3 results/e0_restore/frozen3_v2_shard*.json \
      --iphi data/insc_data/raw/iphi.jsonl --out results/analysis/restoration_v2_fold3.json

Reports
  overall / by gap length L: CER, top-1, top-20, letters-only CER
  error decomposition of wrong top-1s: word-division-only | letters wrong but gold in top-20 (ranking)
      | gold not in top-20 (recall); edit-operation mix (substitution / insertion / deletion)
  by region (PHI main_region) and by date (century of tpq/taq midpoint), with n
  per-document spread (docs with >= 5 gaps)
  calibration: p(top-1) = softmax over the beam of (length-normalized score x L); ECE (10 bins),
      Brier score, AUROC of p for top-1 correctness; risk-coverage (top-1 accuracy when answering
      only the most confident c% of gaps, c = 100, 90, ..., 10) and AURC
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np


def lev_ops(a, b):
    """Levenshtein distance + counts of (sub, ins, del) on one optimal path."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, sub, ins, dele = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            sub += a[i - 1] != b[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            dele += 1; i -= 1          # char of prediction deleted
        else:
            ins += 1; j -= 1
    return int(D[n, m]), sub, ins, dele


def century(tpq, taq):
    try:
        a = float(tpq) if tpq not in (None, "") else None
        b = float(taq) if taq not in (None, "") else None
    except ValueError:
        return "unknown"
    if a is None and b is None:
        return "unknown"
    mid = (a if b is None else b) if a is None else (a if b is None else (a + b) / 2)
    c = int(math.floor(mid / 100))
    return f"{-c}c BCE" if mid < 0 else f"{c + 1}c CE"


def auroc(scores, labels):
    s = np.asarray(scores); y = np.asarray(labels).astype(bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def summarize(rows):
    if not rows:
        return dict(n=0)
    return dict(n=len(rows), CER=round(float(np.mean([r["cer"] for r in rows])) * 100, 2),
                top1=round(float(np.mean([r["t1"] for r in rows])) * 100, 2),
                top20=round(float(np.mean([r["t20"] for r in rows])) * 100, 2),
                CER_letters=round(float(np.mean([r["cer_let"] for r in rows])) * 100, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--iphi", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    samples = []
    for p in a.inputs:
        j = json.loads(Path(p).read_text())
        if not j.get("complete"):
            print(f"WARNING: {p} incomplete ({j.get('n_done')}/{j.get('n_total')})")
        samples.extend(j["samples"])
    meta = {}
    for line in open(a.iphi, encoding="utf-8"):
        r = json.loads(line)
        meta[int(r["PHI_ID"])] = (r.get("main_region") or "unknown", century(r.get("tpq"), r.get("taq")))

    rows = []
    for s in samples:
        gold, preds, scores, L = s["gold"], s["preds"], s["scores"], s["L"]
        pred = preds[0] if preds else ""
        d, sub, ins, dele = lev_ops(pred, gold)
        gl = gold.replace(" ", ""); pl = pred.replace(" ", "")
        dl = lev_ops(pl, gl)[0]
        t1 = pred == gold; t20 = gold in preds
        if scores:
            z = np.array(scores) * L
            p = np.exp(z - z.max()); p /= p.sum(); conf = float(p[0])
            margin = float(scores[0] - scores[1]) if len(scores) > 1 else float("inf")
        else:
            conf, margin = 0.0, 0.0
        kind = ("correct" if t1 else "division_only" if pl == gl else
                "ranking" if t20 else "recall")
        reg, cen = meta.get(int(s["phi_id"]), ("unknown", "unknown"))
        rows.append(dict(phi=int(s["phi_id"]), L=L, cer=d / max(len(gold), 1), cer_let=dl / max(len(gl), 1),
                         t1=t1, t20=t20, conf=conf, margin=margin, kind=kind, sub=sub, ins=ins, dele=dele,
                         region=reg, century=cen))

    rep = dict(name=a.name, overall=summarize(rows))
    rep["by_L"] = {L: summarize([r for r in rows if r["L"] == L]) for L in sorted({r["L"] for r in rows})}
    wrong = [r for r in rows if not r["t1"]]
    rep["error_kinds_pct_of_all"] = {k: round(100 * sum(r["kind"] == k for r in rows) / len(rows), 2)
                                     for k in ("correct", "division_only", "ranking", "recall")}
    tot_ops = sum(r["sub"] + r["ins"] + r["dele"] for r in wrong) or 1
    rep["edit_ops_pct"] = dict(substitution=round(100 * sum(r["sub"] for r in wrong) / tot_ops, 1),
                               insertion=round(100 * sum(r["ins"] for r in wrong) / tot_ops, 1),
                               deletion=round(100 * sum(r["dele"] for r in wrong) / tot_ops, 1))
    for key in ("region", "century"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        rep[f"by_{key}"] = {g: summarize(v) for g, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))}
    docs = defaultdict(list)
    for r in rows:
        docs[r["phi"]].append(r)
    big = [v for v in docs.values() if len(v) >= 5]
    if big:
        acc = [np.mean([r["t1"] for r in v]) for v in big]
        rep["per_document"] = dict(n_docs=len(docs), n_docs_ge5=len(big),
                                   top1_pct_quartiles=[round(float(q) * 100, 1) for q in np.percentile(acc, [25, 50, 75])])
    # calibration / selective prediction
    conf = np.array([r["conf"] for r in rows]); y = np.array([r["t1"] for r in rows], float)
    bins = np.linspace(0, 1, 11); ece = 0.0; table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - y[m].mean())
            table.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()), conf=round(float(conf[m].mean()), 3),
                              acc=round(float(y[m].mean()), 3)))
    order = np.argsort(-conf)
    rc = []
    for c in range(100, 0, -10):
        k = max(1, int(round(len(rows) * c / 100)))
        rc.append(dict(coverage=c, top1=round(float(y[order[:k]].mean()) * 100, 2)))
    cum = np.cumsum(1 - y[order]) / np.arange(1, len(rows) + 1)
    rep["calibration"] = dict(ece=round(float(ece), 4), brier=round(float(np.mean((conf - y) ** 2)), 4),
                              auroc_conf=auroc(conf, y), auroc_margin=auroc([r["margin"] for r in rows], y),
                              reliability=table, risk_coverage=rc, aurc=round(float(cum.mean()), 4))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(dict(name=a.name, overall=rep["overall"], kinds=rep["error_kinds_pct_of_all"],
                          ece=rep["calibration"]["ece"], auroc=rep["calibration"]["auroc_conf"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
