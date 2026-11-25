"""Generate configs/folds/fold_{0..9}.json — byte-identical to the flagship config
except name and out_dir. Data paths are NOT in the config: each fold job exports
STOICHEIA_DATA_ROOT=$STOICHEIA_DATA/folds/fold_k so stage_shards.sh points STOICHEIA_DATA at the fold."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]   # repo root (scripts/pretrain/ -> ../..)
base = json.load(open(root / "configs" / "pretrain" / "stoicheia.json"))
out = root / "configs" / "pretrain" / "folds"
out.mkdir(parents=True, exist_ok=True)
for k in range(10):
    c = dict(base)
    c["name"] = f"stoicheia_fold_{k}"
    c["out_dir"] = f"$STOICHEIA_DATA/runs/stoicheia_fold_{k}"
    # dev-driven regime: eval (anneal stall / early stop / best.pt) runs on the fold's
    # REAL val split (unseen works); no intra-train holdout is set aside.
    c["eval_shards"] = f"$STOICHEIA_DATA/folds/fold_{k}/val_shards/v1_punct"
    c["train_holdout"] = False
    # both stages run until dev stall, not until a schedule cap: stable phase has an
    # unreachable hard cap; decay window doubled (48k) with dev-stall early-stop as the
    # practical terminator (anneal_patience=3 evals, active from mid-decay)
    c["hard_max_steps"] = 1000000
    c["decay_frac"] = 0.4
    # staged anneal (proven: staged branch hit bpc 0.235 vs classic hard-switch 0.338
    # on the flagship): full mix for the first half of the decay, drop bronze at 50%,
    # gold-only for the last 10% — avoids the +0.04 hard-switch transient at near-peak LR
    c["anneal_phases"] = [[0.5, [1.0, 1.0, 0.3]], [0.9, [1.0, 1.0, 0.0]],
                          [1.0, [1.0, 0.0, 0.0]]]
    json.dump(c, open(out / f"fold_{k}.json", "w"), indent=2)
    print(out / f"fold_{k}.json")
