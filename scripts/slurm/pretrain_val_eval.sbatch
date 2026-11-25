#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH --job-name=gcb-valeval
#SBATCH -o logs/valeval-%j.out
#SBATCH -e logs/valeval-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Score checkpoints on the fold's REAL val split (unseen works). Tiny job (1 GPU) —
# schedules as backfill even when the training fleet saturates the account cap.
# Usage: sbatch scripts/slurm/pretrain_val_eval.sbatch <fold 0-9> [ckpt_name=best.pt]
set -euo pipefail
FOLD="${1:?usage: sbatch scripts/val_eval.sbatch <fold 0-9> [ckpt]}"
CKPT="${2:-best.pt}"
STOICHEIA_ROOT=$STOICHEIA_ROOT
source $STOICHEIA_ROOT/env.sh

CK=$STOICHEIA_DATA/runs/stoicheia_fold_$FOLD/$CKPT
VS=$STOICHEIA_DATA/folds/fold_$FOLD/val_shards/v1_punct
[ -f "$CK" ] || { echo "missing checkpoint $CK"; exit 1; }
[ -f "$VS/index.parquet" ] || { echo "missing val shards $VS"; exit 1; }

apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
  set -e
  export PYTHONPATH=$STOICHEIA_ROOT
  cd $STOICHEIA_ROOT
  python -m eval.val_eval --ckpt $CK --val-shards $VS --n 1024
"
echo VAL_EVAL_DONE
