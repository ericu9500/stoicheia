#!/bin/bash
# Submit the final run as a CHAIN of dependent jobs instead of one long monolithic request.
# Each job requests a modest wall-time (schedules far more easily as backfill), checkpoints
# every ckpt_every steps, and the trainer auto-resumes from out_dir/last.pt — so a crash or
# timeout only loses at most one checkpoint interval, not the whole run.
#
# Usage: scripts/pretrain/launch_chain.sh [n_jobs] [hours_per_job] [config]
#   scripts/pretrain/launch_chain.sh 6 8 configs/pretrain/stoicheia.json   # 6 x 8h = 48h budget
set -euo pipefail
N=${1:-6}
HOURS=${2:-8}
CONFIG=${3:-configs/pretrain/stoicheia.json}
cd "$(dirname "$0")/../.."

PREV=""
JOBIDS=()
for i in $(seq 1 "$N"); do
  if [ -z "$PREV" ]; then DEP=""; else DEP="--dependency=afterany:$PREV"; fi
  J=$(sbatch --parsable --time="${HOURS}:00:00" --job-name=gcb-final $DEP \
    -o "logs/final-%j.out" -e "logs/final-%j.out" \
    scripts/slurm/pretrain_final.sbatch "$CONFIG")
  echo "chain job $i/$N: $J (dep: ${PREV:-none})"
  JOBIDS+=("$J")
  PREV=$J
done
echo "chain: ${JOBIDS[*]}"
echo "total wall-time budget: $((N * HOURS))h"
