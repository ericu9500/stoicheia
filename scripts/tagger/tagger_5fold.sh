#!/bin/bash
# Launch the final recipe on all 5 folds (one 4-GPU job each).
# Usage: scripts/tagger_5fold.sh [config] [ckpt]
set -euo pipefail
TAGGER_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${1:-configs/tagger/tagger_final.json}
CKPT=${2:-}
cd "$TAGGER_ROOT"
source env.sh
for FOLD in 0 1 2 3 4; do
  GEN="configs/tagger/_gen_fold${FOLD}.json"
  python - "$CONFIG" "$FOLD" "$GEN" <<'EOF'
import json, re, sys
cfg = json.load(open(sys.argv[1]))
fold = int(sys.argv[2])
cfg["fold"] = fold
cfg["out_dir"] = re.sub(r"(_fold\d+)?$", f"_fold{fold}", cfg["out_dir"], count=1)
json.dump(cfg, open(sys.argv[3], "w"), indent=2)
print(sys.argv[3], "->", cfg["out_dir"])
EOF
  if [ -n "$CKPT" ]; then
    sbatch scripts/slurm/tagger_tagger.sbatch "$GEN" --ckpt "$CKPT"
  else
    sbatch scripts/slurm/tagger_tagger.sbatch "$GEN"
  fi
done
echo "submitted 5 folds; evaluate each with:"
echo "  python -m tagger.evaluate --run \$STOICHEIA_DATA/runs/<out_dir> --gold \$STOICHEIA_DATA/treebanks/oga_repo/kfold/test.conllu"
