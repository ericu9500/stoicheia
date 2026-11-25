#!/bin/bash
# Final matrix: 5 folds x 2 seeds from a winning recipe config.
# Usage: scripts/launch_finals.sh <winning_config.json>
set -euo pipefail
TAGGER_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${1:?usage: scripts/launch_finals.sh <config.json>}
cd "$TAGGER_ROOT"
source env.sh
for FOLD in 0 1 2 3 4; do
  for SEED in 0 1; do
    GEN="configs/tagger/_final_f${FOLD}_s${SEED}.json"
    python - "$CONFIG" "$FOLD" "$SEED" "$GEN" <<'EOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
fold, seed = int(sys.argv[2]), int(sys.argv[3])
cfg["fold"] = fold
cfg["seed"] = seed
cfg["name"] = f"tagger_final_f{fold}_s{seed}"
cfg["out_dir"] = f"$STOICHEIA_DATA/runs/tagger_final_f{fold}_s{seed}"
json.dump(cfg, open(sys.argv[4], "w"), indent=2)
EOF
    J=$(sbatch --parsable scripts/slurm/tagger_tagger.sbatch "$GEN")
    sbatch --dependency=afterany:$J scripts/slurm/tagger_eval.sbatch \
      "\$STOICHEIA_DATA/runs/tagger_final_f${FOLD}_s${SEED}" \
      "\$STOICHEIA_DATA/treebanks/oga_repo/kfold/test.conllu" > /dev/null
    echo "fold $FOLD seed $SEED: train $J (+chained test eval)"
  done
done
