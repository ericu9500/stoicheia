#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH --job-name=insc-eval
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
# Usage: sbatch scripts/slurm/insc_eval.sbatch <ckpt> <split> <n_per_length> <out.json>
set -euo pipefail
CKPT="${1:?ckpt}"; SPLIT="${2:-val}"; N="${3:-200}"; OUT="${4:-}"; EXCL="${5:-}"
INS_ROOT=$STOICHEIA_ROOT

# Sourced ONCE here, at the top-level sbatch shell -- see the finetune launchers for why
# (re-sourcing env.sh inside a remote srun'd `bash -lc` can silently leave $SIF unset).
source "$INS_ROOT/env.sh"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    cd $INS_ROOT
    python insc/eval/restore.py --ckpt $CKPT --split $SPLIT --n $N ${OUT:+--out $OUT} ${EXCL:+--exclude $EXCL}
  "
echo EVAL_JOB_END
