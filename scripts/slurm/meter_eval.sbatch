#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --job-name=gcb-meter-eval
#SBATCH -o logs/eval-%j.out
#SBATCH -e logs/eval-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Usage: sbatch scripts/slurm/meter_eval.sbatch <best.pt> [predict args...]
set -euo pipefail
MODEL="${1:?usage: sbatch scripts/eval.sbatch <best.pt> [args]}"
shift || true
EXTRA="${*:---scan-split --norma}"
METER_ROOT=$STOICHEIA_ROOT
source $METER_ROOT/env.sh
echo "model=$MODEL extra='$EXTRA' $(date)"

apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
  set -e
  export PYTHONPATH=$METER_ROOT:$STOICHEIA_ROOT
  export METER_DATA=$METER_DATA STOICHEIA_DATA=$STOICHEIA_DATA MACRONIZER_SRC=$MACRONIZER_SRC
  cd $METER_ROOT
  python -m meter.predict --model $MODEL --micro 32 $EXTRA
"
echo "EVAL FINISHED $(date)"
