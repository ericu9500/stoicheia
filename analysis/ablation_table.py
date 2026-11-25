"""Collect every finished ablation run into one table (E2-E4) with paired tests against the reference arm.

  python analysis/ablation_table.py --out results/ablation_table.json [--md results/ablation_table.md]
Parsing: test LAS/UAS/XPOS/lemma per fold (runs/parse/<arm>_f<k>_s<s>/test_scores_greedy.json), paired by
fold with planes_ref_s0 seed 0 -> exact sign-flip p over the 5 folds (floor 0.0625).
Macronization: Norma test balanced accuracy per seed (runs/meter/<arm>_mac_s<s>/norma_scores.txt).
Restoration: frozen fold-3 strict CER/top-1/top-20 per fine-tune seed (runs/insc/<arm>_t3_s<s>/strict_shard*.json,
merged by sample), paired per sample vs reference with McNemar on top-1.
Incomplete arms are listed with what is missing.
"""
import argparse, glob, itertools, json, math, re
from collections import defaultdict
import os
from pathlib import Path

R = Path(os.environ["STOICHEIA_DATA"])
REF = "planes_ref_s0"


def signflip(d):
    d = [x for x in d if x is not None]
    if not d:
        return None
    m = abs(sum(d) / len(d)); n = 0; hit = 0
    for s in itertools.product((1, -1), repeat=len(d)):
        n += 1; hit += abs(sum(a * b for a, b in zip(s, d)) / len(d)) >= m - 1e-12
    return hit / n


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def parse_scores():
    out = defaultdict(dict)
    for p in glob.glob(str(R / "runs/ablate/parse/*/test_scores_greedy.json")):
        m = re.match(r"(.+)_f(\d)_s(\d)$", Path(p).parent.name)
        j = json.load(open(p))
        out[m.group(1)][(int(m.group(2)), int(m.group(3)))] = {k: 100 * j[k] for k in ("las", "uas", "xpos_exact", "lemma_script", "upos")}
    return out


def meter_scores():
    out = defaultdict(dict)
    for p in glob.glob(str(R / "runs/ablate/meter/*/norma_scores.txt")):
        m = re.match(r"(.+)_mac_s(\d)$", Path(p).parent.name) or re.match(r"(mac_v2_docclean)_s(\d)$", Path(p).parent.name)
        if not m:
            continue
        for line in open(p):
            if line.startswith("macron test:"):
                j = json.loads(line.split(":", 1)[1])
                out[m.group(1)][int(m.group(2))] = dict(bal_acc=100 * j.get("bal_acc", float("nan")), acc=100 * j.get("acc", float("nan")))
    return out


def insc_scores():
    out = defaultdict(dict)
    for d in glob.glob(str(R / "runs/ablate/insc/*_t3_s*")):
        m = re.match(r"(.+)_t3_s(\d)$", Path(d).name)
        shards = sorted(glob.glob(f"{d}/strict_shard*.json"))
        js = [json.load(open(p)) for p in shards]
        if len(js) < 4 or not all(j.get("complete") for j in js):
            continue
        samples = {(s["phi_id"], s["L"], s["start"]): s for j in js for s in j["samples"]}
        out[m.group(1)][int(m.group(2))] = samples
    return out


def summarize_insc(samples):
    import numpy as np
    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = cur
        return prev[-1]
    cer = [lev(s["preds"][0] if s["preds"] else "", s["gold"]) / len(s["gold"]) for s in samples.values()]
    return dict(n=len(cer), CER=100 * float(np.mean(cer)),
                top1=100 * float(np.mean([bool(s["preds"]) and s["preds"][0] == s["gold"] for s in samples.values()])),
                top20=100 * float(np.mean([s["gold"] in s["preds"] for s in samples.values()])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--md")
    a = ap.parse_args()
    import numpy as np
    P, M, I = parse_scores(), meter_scores(), insc_scores()
    arms = sorted(set(P) | set(M) | set(I) | {Path(p).stem for p in glob.glob(str(Path(__file__).resolve().parent.parent / "configs/ablate/pretrain/*.json"))} | {"planes_rand"})
    rows = []
    for arm in arms:
        r = dict(arm=arm)
        s0 = {f: v for (f, s), v in P.get(arm, {}).items() if s == 0}
        if s0:
            r["parse_n"] = len(s0)
            for k in ("las", "uas", "xpos_exact", "lemma_script"):
                r[f"parse_{k}"] = round(float(np.mean([v[k] for v in s0.values()])), 2)
            ref = {f: v for (f, s), v in P.get(REF, {}).items() if s == 0}
            common = sorted(set(s0) & set(ref))
            if arm != REF and common:
                d = [s0[f]["las"] - ref[f]["las"] for f in common]
                r["parse_las_delta_vs_ref"] = round(float(np.mean(d)), 2); r["parse_las_p"] = signflip(d)
        if M.get(arm):
            v = M[arm]
            r["mac_n"] = len(v); r["mac_bal_acc"] = round(float(np.mean([x["bal_acc"] for x in v.values()])), 2)
            r["mac_bal_acc_sd"] = round(float(np.std([x["bal_acc"] for x in v.values()], ddof=1)), 2) if len(v) > 1 else None
        if I.get(arm):
            summ = [summarize_insc(v) for v in I[arm].values()]
            r["insc_n_seeds"] = len(summ)
            for k in ("CER", "top1", "top20"):
                r[f"insc_{k}"] = round(float(np.mean([x[k] for x in summ])), 2)
            if arm != REF and I.get(REF) and 0 in I[arm] and 0 in I[REF]:
                A, B = I[arm][0], I[REF][0]; keys = set(A) & set(B)
                ok = lambda s: bool(s["preds"]) and s["preds"][0] == s["gold"]
                b = sum(ok(A[k]) and not ok(B[k]) for k in keys); c = sum(ok(B[k]) and not ok(A[k]) for k in keys)
                r["insc_top1_only_arm"], r["insc_top1_only_ref"], r["insc_mcnemar_p"] = b, c, mcnemar(b, c)
        rows.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    cols = ["arm", "parse_n", "parse_las", "parse_las_delta_vs_ref", "parse_las_p", "parse_xpos_exact", "parse_lemma_script",
            "mac_n", "mac_bal_acc", "mac_bal_acc_sd", "insc_n_seeds", "insc_CER", "insc_top1", "insc_top20", "insc_mcnemar_p"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    md = "\n".join(lines)
    if a.md:
        Path(a.md).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
