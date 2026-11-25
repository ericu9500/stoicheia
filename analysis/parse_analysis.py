"""E5 (parsing): LAS/UAS breakdowns from parse_dump.py outputs, optionally paired against a second system.

  python analysis/parse_analysis.py results/e5/parse_released_test.jsonl [--vs other.jsonl] --out x.json
Breakdowns: by gold deprel, by gold UPOS of the dependent, by sentence length band, by gold
dependency distance band, root attachment; error types among wrong arcs (head wrong | label only).
With --vs: paired arc-level differences (only arcs present in both dumps), McNemar exact p on LAS.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path


def load(p):
    return {(r["sent"], r["word"]): r for r in map(json.loads, open(p, encoding="utf-8"))}


def band(x, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= x < hi:
            return f"{lo}-{hi - 1}"
    return f">={edges[-1]}"


def stats(rs):
    n = len(rs)
    if not n:
        return dict(n=0)
    uas = sum(r["gold_head"] == r["pred_head"] for r in rs); las = sum(r["gold_head"] == r["pred_head"] and r["gold_rel"] == r["pred_rel"] for r in rs)
    return dict(n=n, UAS=round(100 * uas / n, 2), LAS=round(100 * las / n, 2))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def breakdown(d):
    rs = list(d.values())
    out = dict(overall=stats(rs))
    keys = dict(deprel=lambda r: r["gold_rel"], upos=lambda r: r["upos"],
                sent_len=lambda r: band(r["sent_len"], [1, 6, 11, 21, 41]),
                distance=lambda r: "root" if r["gold_head"] == 0 else band(abs(r["gold_head"] - r["word"]), [1, 2, 3, 5, 8, 13]))
    for name, f in keys.items():
        g = defaultdict(list)
        for r in rs:
            g[f(r)].append(r)
        out[f"by_{name}"] = {k: stats(v) for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))}
    wrong = [r for r in rs if not (r["gold_head"] == r["pred_head"] and r["gold_rel"] == r["pred_rel"])]
    out["wrong_arcs"] = dict(n=len(wrong), head_wrong=sum(r["gold_head"] != r["pred_head"] for r in wrong),
                             label_only=sum(r["gold_head"] == r["pred_head"] for r in wrong))
    conf = defaultdict(int)
    for r in wrong:
        if r["gold_head"] == r["pred_head"]:
            conf[f"{r['gold_rel']}->{r['pred_rel']}"] += 1
    out["top_label_confusions"] = sorted(conf.items(), key=lambda kv: -kv[1])[:15]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--vs"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = load(a.dump)
    rep = dict(system=a.dump, analysis=breakdown(A))
    if a.vs:
        B = load(a.vs)
        keys = sorted(set(A) & set(B))
        ok = lambda r: r["gold_head"] == r["pred_head"] and r["gold_rel"] == r["pred_rel"]
        b = sum(ok(A[k]) and not ok(B[k]) for k in keys); c = sum(ok(B[k]) and not ok(A[k]) for k in keys)
        rep["paired"] = dict(other=a.vs, n_common=len(keys), only_this_correct=b, only_other_correct=c,
                             mcnemar_p=mcnemar(b, c), other_analysis=breakdown({k: B[k] for k in keys}))
        d = defaultdict(lambda: [0, 0, 0])
        for k in keys:
            rel = A[k]["gold_rel"]; d[rel][0] += 1; d[rel][1] += ok(A[k]); d[rel][2] += ok(B[k])
        rep["paired"]["las_delta_by_deprel"] = {r: dict(n=v[0], delta=round(100 * (v[1] - v[2]) / v[0], 2))
                                                for r, v in sorted(d.items(), key=lambda kv: -kv[1][0])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep["analysis"]["overall"]), json.dumps({k: v for k, v in rep.get("paired", {}).items() if k in ("only_this_correct", "only_other_correct", "mcnemar_p")}))


if __name__ == "__main__":
    main()
