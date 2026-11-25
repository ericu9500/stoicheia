"""Merge sharded ptp_baseline.py outputs for the Aeneas-Greek (Nature 2025) rows.

Strict shards are per-length files (strict_L{L}_s{i}.json); recent shards are
stride slices (recent_s{i}.json). Both are merged per length with sample-count
weighting, then summarized the same two ways as the existing tables:
  * strict: AVG over L=1..10 of per-length means (the convention of the
    strict table) -- comparable to the Ithaca/Stoicheia rows there.
  * recent: macro over lengths + sample-weighted micro, comparable to the
    uncontaminated table rows.

  python3 analysis/merge_ptp.py
"""
import glob, json, os

D = os.path.expandvars("$STOICHEIA_ROOT/ptp_out")


def merge(pattern):
    per = {}
    err = 0
    for f in sorted(glob.glob(f"{D}/{pattern}")):
        d = json.load(open(f))
        err += d.get("errors", 0)
        for L, v in d.get("per_L", {}).items():
            a = per.setdefault(int(L), dict(n=0, cer=0.0, t1=0.0, t20=0.0))
            a["n"] += v["n"]
            a["cer"] += v["CER"] * v["n"]
            a["t1"] += v["top1"] * v["n"]
            a["t20"] += v["top20"] * v["n"]
    if not per:
        return None
    by = {L: dict(n=a["n"], CER=a["cer"] / a["n"], top1=a["t1"] / a["n"],
                  top20=a["t20"] / a["n"]) for L, a in sorted(per.items())}
    n = sum(v["n"] for v in by.values())
    return dict(errors=err, n=n,
                CER_macro=sum(v["CER"] for v in by.values()) / len(by),
                top1=sum(v["top1"] * v["n"] for v in by.values()) / n,
                top20=sum(v["top20"] * v["n"] for v in by.values()) / n,
                by_length=by)


for name, pat, ref in (
    ("STRICT (vs Table 1: Ithaca 24.55/63.0/74.9, Stoicheia v4 15.46/74.5/81.7)",
     "strict_L*_s*.json", 3000),
    ("RECENT INSCRIPTIONS (vs Table: Ithaca 31.18/48.41/60.67, "
     "Llama 28.59/51.28/68.62, Stoicheia 20.20/64.41/73.34)",
     "recent_s*.json", 4111),
):
    m = merge(pat)
    print("=" * 74)
    print("AENEAS-GREEK 2025 --", name)
    if not m:
        print("  (no output yet)")
        continue
    done = m["n"]
    print(f"  n={done}/{ref}  errors={m['errors']}")
    print(f"  CER={m['CER_macro']:.4f}  top1={m['top1']:.4f}  top20={m['top20']:.4f}")
    for L, v in m["by_length"].items():
        print(f"    L={L:<3} n={v['n']:<4} CER={v['CER']:.4f} "
              f"top1={v['top1']:.4f} top20={v['top20']:.4f}")
