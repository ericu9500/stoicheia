#!/bin/bash
# Launch all 10 fold chains with at most WIDTH chains in flight, pure SLURM dependencies:
# fold k's chain head waits (afterany) on the TAIL of fold (k-WIDTH)'s chain, so the
# pipeline self-advances with no babysitting daemon. afterany (not afterok) means a
# crashed fold never deadlocks the rest — resume it with launch_fold_chain.sh <k>.
#
# Usage: scripts/pretrain/launch_all_folds.sh [width=3] [n_jobs=6] [hours=8]
set -euo pipefail
WIDTH=${1:-3}
N=${2:-6}
HOURS=${3:-5}
cd "$(dirname "$0")"

declare -a TAIL
for k in 0 1 2 3 4 5 6 7 8 9; do
  DEP=""
  if [ "$k" -ge "$WIDTH" ]; then DEP=${TAIL[$((k - WIDTH))]}; fi
  TAIL[$k]=$(./launch_fold_chain.sh "$k" "$N" "$HOURS" "$DEP" | tail -1)
  echo "fold $k: tail job ${TAIL[$k]} (head gated on: ${DEP:-none})"
done
echo "all 10 fold chains submitted ($((10 * N)) jobs, width $WIDTH)"
