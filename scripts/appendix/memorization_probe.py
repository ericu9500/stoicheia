"""E8: does the fold decontamination matter? Memorization gap of the released fold models.

For every fold k, sample pristine records belonging to fold k's TEST works (data/fold_manifests).
Score each record's masked-letter bits/char (eval/intrinsic.py protocol: iid masking at t=0.30,
boundary/diacritic/punctuation channels unknown) with
  - fold k's model (never saw the work),
  - two other fold models j for which the work lies in the TRAINING zone
    (j not in {k, k-1}: fold j's val bucket is k-1+1... i.e. work is val for fold k-1),
paired per record. The difference seen - unseen is the memorization advantage a model has on
text it trained on; the rotation exists so that an editor can always use the unseen model.

  python scripts/appendix/memorization_probe.py --hub-pt data/hub_pt --shards data/shards/v1_punct \
      --manifests code/data/fold_manifests --n-per-fold 60 --out results/e8_memorization.json
"""
from __future__ import annotations

import argparse, gzip, json, math, os, subprocess, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.intrinsic import evaluate, load_model


def group_key(source, rid):
    if source == "oga":
        return "oga:" + ".".join(rid.split(".")[:2])
    return f"{source}:{rid.split('#')[0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True, help="dir with Stoicheia-fold-k snapshots")
    ap.add_argument("--hub-pt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--manifests", required=True)
    ap.add_argument("--n-per-fold", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = pq.read_table(Path(a.shards) / "index.parquet", columns=["offset", "length", "tier", "source", "id"])
    offs = idx.column("offset").to_numpy(); lens = idx.column("length").to_numpy()
    tier = idx.column("tier").to_numpy(zero_copy_only=False)
    src = idx.column("source").to_numpy(zero_copy_only=False); rid = idx.column("id").to_numpy(zero_copy_only=False)
    ok = np.flatnonzero((tier == "pristine") & (lens >= 256) & (lens <= 1024))
    keys = np.array([group_key(src[i], rid[i]) for i in ok], dtype=object)
    planes = {p: np.memmap(Path(a.shards) / f"{p}.bin", dtype=np.uint8, mode="r")
              for p in ("chars", "boundary", "dia", "cap", "punct")}
    rng = np.random.default_rng(0)
    picks = {}
    for k in range(10):
        with gzip.open(Path(a.manifests) / f"fold_{k}_test_works.tsv.gz", "rt") as f:
            groups = {l.split("\t")[0] for l in f.read().splitlines()[1:]
                      if l.split("\t")[1] == "literary"}
        cand = ok[np.isin(keys, list(groups))]
        # at most 3 records per work so a few big works cannot dominate
        by = {}
        for i in rng.permutation(cand):
            g = group_key(src[i], rid[i])
            if len(by.setdefault(g, [])) < 3:
                by[g].append(int(i))
        flat = [i for v in by.values() for i in v]
        rng.shuffle(flat)
        picks[k] = flat[:a.n_per_fold]
        print(f"fold {k}: {len(groups)} literary test works, {len(cand)} candidate records, {len(picks[k])} picked", flush=True)

    def recs(ids):
        return [{p: np.array(planes[p][int(offs[i]):int(offs[i] + lens[i])]) for p in planes} for i in ids]

    scores = {}   # (k, model_fold) -> per-record bpc
    for m in range(10):
        pt = Path(a.hub_pt) / f"fold-{m}.pt"
        if not pt.exists():
            subprocess.run([sys.executable, str(Path(__file__).parent / "hf_local_to_pt.py"),
                            str(Path(a.hub) / f"Stoicheia-fold-{m}"), str(pt)], check=True)
        model, _ = load_model(str(pt), device)
        model.cfg.attn_impl = "sdpa"
        for k in range(10):
            unseen = m == k
            train_fold = m not in (k, (k - 1) % 10)       # fold k-1's val bucket is k
            if not (unseen or m in ((k + 3) % 10, (k + 6) % 10)):
                continue
            assert unseen or train_fold
            per = [evaluate(model, [r], device, T=1024, micro=1)["bits_per_char"] for r in recs(picks[k])]
            scores[f"{k}:{m}"] = per
            print(f"records of fold {k} scored by fold-{m} model ({'UNSEEN' if unseen else 'seen'}): "
                  f"mean bpc {np.mean(per):.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    rows = []
    for k in range(10):
        u = np.array(scores[f"{k}:{k}"])
        s = (np.array(scores[f"{k}:{(k+3)%10}"]) + np.array(scores[f"{k}:{(k+6)%10}"])) / 2
        rows.append(dict(fold=k, n=len(u), bpc_unseen=round(float(u.mean()), 4), bpc_seen=round(float(s.mean()), 4),
                         gap=round(float((u - s).mean()), 4), frac_records_seen_lower=round(float((s < u).mean()), 3)))
    d = np.array([r["gap"] for r in rows])
    signs = np.array([[1 if (b >> i) & 1 else -1 for i in range(10)] for b in range(1024)])
    p = float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())
    res = dict(per_fold=rows, mean_gap=round(float(d.mean()), 4), folds_with_positive_gap=int((d > 0).sum()),
               signflip_p=p, rel_gap_pct=round(100 * float(d.mean() / np.mean([r["bpc_seen"] for r in rows])), 1),
               note="gap = bpc(unseen model) - bpc(models that trained on the work); positive = memorization advantage")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=res, picks={k: [str(rid[i]) for i in v] for k, v in picks.items()},
                                           scores=scores), indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
