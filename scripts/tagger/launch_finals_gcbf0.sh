#!/bin/bash
# Final 5x2 matrix REBASED on the stoicheia_fold_0 torso (decontaminated pretraining) —
# identical recipe to launch_finals.sh, distinct run names so the flagship-based
# tagger_final_* runs are untouched. Comparison target: XPOS 94.22 / lemma 94.41 /
# UPOS 97.30 (flagship staged-anneal torso, n=10).
# Usage: scripts/launch_finals_gcbf0.sh <winning_config.json>
set -euo pipefail
TAGGER_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${1:?usage: scripts/launch_finals_gcbf0.sh <config.json>}
cd "$TAGGER_ROOT"
source env.sh
TORSO=$STOICHEIA_DATA/runs/stoicheia_fold_0/best.pt
for FOLD in 0 1 2 3 4; do
  for SEED in 0 1; do
    GEN="configs/tagger/_gcbf0_f${FOLD}_s${SEED}.json"
    python - "$CONFIG" "$FOLD" "$SEED" "$GEN" "$TORSO" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
fold, seed = int(sys.argv[2]), int(sys.argv[3])
cfg["fold"] = fold
cfg["seed"] = seed
cfg["ckpt"] = sys.argv[5]
cfg["name"] = f"tagger_gcbf0_f{fold}_s{seed}"
cfg["out_dir"] = f"$STOICHEIA_DATA/runs/tagger_gcbf0_f{fold}_s{seed}"
json.dump(cfg, open(sys.argv[4], "w"), indent=2)
EOF
    J=$(sbatch --parsable --time=01:30:00 scripts/slurm/tagger_tagger.sbatch "$GEN")
    sbatch --dependency=afterany:$J --time=01:00:00 scripts/slurm/tagger_eval.sbatch \
      "\$STOICHEIA_DATA/runs/tagger_gcbf0_f${FOLD}_s${SEED}" \
      "\$STOICHEIA_DATA/treebanks/oga_repo/kfold/test.conllu" > /dev/null
    echo "fold $FOLD seed $SEED: train $J (+chained test eval)"
  done
done
