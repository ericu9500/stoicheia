#!/bin/bash
# Submit the documentary-clean run as a chain of dependent jobs.
# Usage: scripts/pretrain/launch_doc_chain.sh [n_jobs=6] [hours=8] [dep_jobid]
# Prints one line per job to stderr; the LAST stdout line is the tail job id.
set -euo pipefail
N=${1:-6}
HOURS=${2:-5}
HEADDEP=${3:-}
cd "$(dirname "$0")/../.."

PREV="$HEADDEP"
JOBIDS=()
for i in $(seq 1 "$N"); do
  if [ -z "$PREV" ]; then DEP=""; else DEP="--dependency=afterany:$PREV"; fi
  J=$(sbatch --parsable --time="${HOURS}:00:00" --job-name="gcb-doc" $DEP \
    -o "logs/doc-%j.out" -e "logs/doc-%j.out" \
    scripts/slurm/pretrain_doc_clean.sbatch)
  echo "doc_clean chain job $i/$N: $J (dep: ${PREV:-none})" >&2
  JOBIDS+=("$J")
  PREV=$J
done
echo "doc_clean chain: ${JOBIDS[*]} (budget $((N * HOURS))h)" >&2
echo "$PREV"
