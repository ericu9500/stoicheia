"""E5/E6 comparative: Stoicheia v2 vs Ithaca vs Aeneas on the identical frozen fold-3 samples.

Joins per-sample outputs by (phi_id, L, start). Reports per system and per L; pairwise McNemar on top-1/top-20;
error overlap (gaps solved by none / only one / all); oracle union top-1; by region and date for all three;
confidence ranking (AUROC of top-1 correctness, top-1 accuracy at 60/80 % coverage) with each system's own
score scale turned into a beam distribution (ours: softmax(length-normalized score x L); Ithaca: softmax of summed
log-probs; Aeneas: scores are probabilities) after merging duplicate strings.
"""
import json, math, sys
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from restoration_analysis import auroc, century, lev_ops

R = Path(os.environ["STOICHEIA_DATA"])


def load(paths):
    out = {}
    for p in paths:
        for s in json.load(open(p))["samples"]:
            out[(int(s["phi_id"]), s["L"], s["start"])] = s
    return out


def conf_of(system, s):
    preds, sc = s["preds"], s["scores"]
    if not preds:
        return 0.0
    if system == "stoicheia":
        z = np.array(sc) * s["L"]; p = np.exp(z - z.max())
    elif system == "ithaca":
        z = np.array(sc, float); p = np.exp(z - z.max())
    else:
        p = np.array(sc, float)
    agg = defaultdict(float)
    for x, w in zip(preds, p):
        agg[x] += w
    tot = sum(agg.values()) or 1.0
    return agg[preds[0]] / tot


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    S = {"stoicheia": load(sorted(R.glob("results/e0_restore/frozen3_v2_shard*.json"))),
         "ithaca": load([R / "results/e0_baselines/ithaca_frozen3.json"]),
         "aeneas": load([R / "results/e0_baselines/aeneas_frozen3.json"])}
    keys = sorted(set.intersection(*(set(v) for v in S.values())))
    meta = {}
    for line in open(R / "data/insc_data/raw/iphi.jsonl", encoding="utf-8"):
        r = json.loads(line); meta[int(r["PHI_ID"])] = (r.get("main_region") or "unknown", century(r.get("tpq"), r.get("taq")))
    gold = {k: S["stoicheia"][k]["gold"] for k in keys}
    t1 = {n: {k: bool(S[n][k]["preds"]) and S[n][k]["preds"][0] == gold[k] for k in keys} for n in S}
    t20 = {n: {k: gold[k] in S[n][k]["preds"] for k in keys} for n in S}
    cer = {n: {k: lev_ops(S[n][k]["preds"][0] if S[n][k]["preds"] else "", gold[k])[0] / len(gold[k]) for k in keys} for n in S}
    rep = dict(n=len(keys), systems={})
    for n in S:
        rep["systems"][n] = dict(CER=round(100 * np.mean(list(cer[n].values())), 2), top1=round(100 * np.mean(list(t1[n].values())), 2),
                                 top20=round(100 * np.mean(list(t20[n].values())), 2))
    rep["by_L"] = {L: {n: dict(CER=round(100 * np.mean([cer[n][k] for k in keys if k[1] == L]), 2),
                               top1=round(100 * np.mean([t1[n][k] for k in keys if k[1] == L]), 2)) for n in S} for L in range(1, 11)}
    pairs = {}
    for a, b in (("stoicheia", "ithaca"), ("stoicheia", "aeneas"), ("aeneas", "ithaca")):
        for m, d in (("top1", t1), ("top20", t20)):
            x = sum(d[a][k] and not d[b][k] for k in keys); y = sum(d[b][k] and not d[a][k] for k in keys)
            pairs[f"{a}_vs_{b}_{m}"] = dict(only_first=x, only_second=y, p=mcnemar(x, y))
    rep["mcnemar"] = pairs
    solved = [sum(t1[n][k] for n in S) for k in keys]
    rep["overlap_top1_pct"] = {"none": round(100 * np.mean([s == 0 for s in solved]), 2),
                               "exactly_one": round(100 * np.mean([s == 1 for s in solved]), 2),
                               "all_three": round(100 * np.mean([s == 3 for s in solved]), 2),
                               "only_stoicheia": round(100 * np.mean([t1["stoicheia"][k] and not t1["ithaca"][k] and not t1["aeneas"][k] for k in keys]), 2),
                               "baselines_but_not_stoicheia": round(100 * np.mean([(t1["ithaca"][k] or t1["aeneas"][k]) and not t1["stoicheia"][k] for k in keys]), 2),
                               "oracle_union": round(100 * np.mean([s > 0 for s in solved]), 2)}
    for key, idx in (("region", 0), ("century", 1)):
        g = defaultdict(list)
        for k in keys:
            g[meta.get(k[0], ("unknown", "unknown"))[idx]].append(k)
        rep[f"by_{key}"] = {grp: dict(n=len(ks), **{n: round(100 * np.mean([t1[n][k] for k in ks]), 1) for n in S})
                            for grp, ks in sorted(g.items(), key=lambda kv: -len(kv[1])) if len(ks) >= 50}
    cal = {}
    for n in S:
        c = np.array([conf_of(n, S[n][k]) for k in keys]); y = np.array([t1[n][k] for k in keys], float)
        o = np.argsort(-c)
        cal[n] = dict(auroc=round(auroc(c, y), 4), **{f"top1_at_{cv}pct": round(100 * float(y[o[:int(len(keys) * cv / 100)]].mean()), 2) for cv in (80, 60, 40)})
    rep["confidence_ranking"] = cal
    out = R / "results/analysis/restoration_compare_frozen3.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("n", "systems", "overlap_top1_pct", "confidence_ranking")}, indent=1, ensure_ascii=False))
    print(json.dumps(rep["mcnemar"], indent=0))
    print(json.dumps(rep["by_L"]))
    print(json.dumps(rep["by_region"], ensure_ascii=False))


if __name__ == "__main__":
    main()
