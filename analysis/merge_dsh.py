"""Merge sharded restore_dsh.py outputs and print the three-way comparison.

Shards are stride slices, so each covers all gap lengths; merging re-weights each
length by its own sample count rather than averaging shard means (which would be
wrong whenever shards see different numbers of samples at a length).

The headline "Overall Average CER" in the DSH 41(1) comparison paper (2026) is the mean of the ten
per-length means, so that is what we report as CER_macro for comparability; the
sample-weighted CER_micro is printed alongside rather than chosen selectively.

  python3 analysis/merge_dsh.py
"""
import glob, json, os

BASE = os.path.join(os.path.dirname(__file__), "dsh")

# Published reference numbers, the DSH 41(1) comparison paper (2026) DSH 41(1) 69-76, recent/unedited sets.
# Inscriptions verified reproducible from their released per-record predictions
# (top-1 51.28%, top-20 68.62% reproduced exactly; CER 0.2859 published).
# Ithaca and papyri rows were scored by us from the released per-record predictions
# using the verified metric port (our calculate_cer reproduces their stored CER with
# zero difference, and their published inscription top-1/top-20 to the decimal).
REF = {
    "inscr": {
        "Llama 3.1 8B (DSH 2026)": dict(CER=0.2859, top1=0.5128, top20=0.6862),
        "Ithaca (Assael et al. 2022)": dict(CER=0.3118, top1=0.4841, top20=0.6067),
    },
    "pap": {
        "Llama 3.1 8B (DSH 2026)": dict(CER=0.3010, top1=0.5037, top20=0.7051),
        # Ithaca is inscriptions-only: no papyri row exists, by design not omission.
    },
}


def merge(pattern):
    per_len, tot, t1, t20 = {}, 0, 0.0, 0.0
    files = sorted(glob.glob(pattern))
    for f in files:
        d = json.load(open(f))
        for L, v in d.get("by_length", {}).items():
            a = per_len.setdefault(int(L), dict(n=0, cer=0.0, t1=0.0, t20=0.0))
            a["n"] += v["n"]
            a["cer"] += v["CER"] * v["n"]      # back out the sum from the mean
            a["t1"] += v["top1"] * v["n"]
            a["t20"] += v["top20"] * v["n"]
        tot += d["n"]; t1 += d["top1"] * d["n"]; t20 += d["top20"] * d["n"]
    if not per_len:
        return None
    by = {L: dict(n=a["n"], CER=a["cer"] / a["n"], top1=a["t1"] / a["n"],
                  top20=a["t20"] / a["n"]) for L, a in sorted(per_len.items())}
    return dict(shards=len(files), n=tot,
                CER_macro=sum(v["CER"] for v in by.values()) / len(by),
                CER_micro=sum(v["CER"] * v["n"] for v in by.values()) / tot,
                top1=t1 / tot, top20=t20 / tot, by_length=by)


for tag, name in (("inscr", "RECENT INSCRIPTIONS"), ("pap", "RECENT PAPYRI")):
    m = merge(f"{BASE}/v4_recent_{tag}_s*.json")
    print("=" * 72); print(name); print("=" * 72)
    if not m:
        print("  (no shard output yet)\n"); continue
    print(f"  shards={m['shards']}  n={m['n']}")
    print(f"  {'system':<32}{'CER':>8}{'top-1':>9}{'top-20':>9}")
    print(f"  {'Stoicheia (ours, 405M)':<32}{m['CER_macro']:>8.4f}{m['top1']:>9.4f}{m['top20']:>9.4f}")
    for k, v in REF[tag].items():
        print(f"  {k:<32}{v['CER']:>8.4f}{v['top1']:>9.4f}{v['top20']:>9.4f}")
    print(f"  (ours sample-weighted CER {m['CER_micro']:.4f}; macro is the comparable one)")
    print("  per length:")
    for L, v in m["by_length"].items():
        print(f"    L={L:<3} n={v['n']:<5} CER={v['CER']:.4f} top1={v['top1']:.4f} top20={v['top20']:.4f}")
    print()
