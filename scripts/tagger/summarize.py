"""Aggregate final-matrix scores: per-run + mean/sd vs published baselines.
Usage: python scripts/summarize.py [runs_glob_prefix]  (default tagger_final_)"""
import glob, json, os, statistics, sys

prefix = sys.argv[1] if len(sys.argv) > 1 else "tagger_final_"
runs_dir = os.path.expandvars("$STOICHEIA_DATA/runs")
rows = []
for d in sorted(glob.glob(f"{runs_dir}/{prefix}*")):
    sf = os.path.join(d, "scores.jsonl")
    if not os.path.exists(sf):
        continue
    for line in open(sf):
        r = json.loads(line)
        if "test.conllu" in r.get("gold", "") and r.get("lexicon") and r.get("constrain"):
            rows.append((os.path.basename(d), r))
seen = {}
for name, r in rows:
    seen[name] = r          # last (most recent) eval per run wins
if not seen:
    print("no test scores found yet")
    sys.exit(0)
print(f"{'run':28s} {'UPOS':>6} {'XPOS':>6} {'Lemma':>6}")
for name, r in sorted(seen.items()):
    print(f"{name:28s} {r['UPOS']:6.2f} {r['XPOS']:6.2f} {r['Lemmas']:6.2f}")
for k in ("UPOS", "XPOS", "Lemmas"):
    v = [r[k] for r in seen.values()]
    m = statistics.mean(v)
    s = statistics.stdev(v) if len(v) > 1 else 0.0
    print(f"{k:8s} mean {m:6.2f}  sd {s:5.2f}  (n={len(v)})")
print("\nbaselines (Celano 2025, test.conllu): Trankit UPOS 96.18 XPOS 91.55 | "
      "GreTa Lemma 91.17 | Dithrax XPOS 90.65 Lemma 86.85")
