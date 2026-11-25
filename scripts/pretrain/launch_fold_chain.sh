#!/bin/bash
# Submit one fold's training as a CHAIN of dependent jobs (same pattern as launch_chain.sh).
#
# Usage: scripts/pretrain/launch_fold_chain.sh <fold 0-9> [n_jobs=6] [hours=8] [dep_jobid]
#   dep_jobid: optional job id the FIRST job of this chain waits on (afterany) — used by
#              launch_all_folds.sh to keep only a few fold chains in flight at once.
# Prints one line per job; the LAST line is the tail job id (capture with `| tail -1`).
set -euo pipefail
FOLD=${1:?usage: launch_fold_chain.sh <fold 0-9> [n_jobs] [hours] [dep_jobid]}
N=${2:-6}
HOURS=${3:-5}
HEADDEP=${4:-}
cd "$(dirname "$0")/../.."

PREV="$HEADDEP"
JOBIDS=()
for i in $(seq 1 "$N"); do
  if [ -z "$PREV" ]; then DEP=""; else DEP="--dependency=afterany:$PREV"; fi
  J=$(sbatch --parsable --time="${HOURS}:00:00" --job-name="gcb-f${FOLD}" $DEP \
    -o "logs/fold${FOLD}-%j.out" -e "logs/fold${FOLD}-%j.out" \
    scripts/slurm/pretrain_fold.sbatch "$FOLD")
  echo "fold $FOLD chain job $i/$N: $J (dep: ${PREV:-none})" >&2
  JOBIDS+=("$J")
  PREV=$J
done
echo "fold $FOLD chain: ${JOBIDS[*]} (budget $((N * HOURS))h)" >&2
echo "$PREV"
