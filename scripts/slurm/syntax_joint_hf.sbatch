#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH --job-name=joint-hf
#SBATCH -o logs/jointhf-%j.out
#SBATCH -e logs/jointhf-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Same-recipe cross-encoder ablation, rough-estimate pass: fold 0, seed 0 only,
# full 4-GPU DDP torchrun per run.
# Usage: sbatch scripts/slurm/syntax_joint_hf.sbatch <encoder_tag>   (e.g. xlmr_base, greberta, agbert, logion, xlmr_large, philberta, mbert)
set -euo pipefail
TAG="${1:?usage: sbatch scripts/joint_hf.sbatch <encoder_tag>}"
SYN_ROOT=$STOICHEIA_ROOT
STOICHEIA_ROOT=$STOICHEIA_ROOT
source $STOICHEIA_ROOT/env.sh
source $SYN_ROOT/env.sh
mkdir -p "$SYN_ROOT/logs"

CFG0="$SYN_ROOT/configs/syntax/joint_${TAG}_f0_s0.json"
[ -f "$CFG0" ] || { echo "missing $CFG0"; exit 1; }

PORT=$((29000 + SLURM_JOB_ID % 1000))

apptainer exec --nv $APPTAINER_BINDS $SIF bash -c "
  set -e
  export PYTHONPATH=$SYN_ROOT:$TAGGER_ROOT:$STOICHEIA_ROOT
  export HF_HOME=$HF_HOME
  cd $SYN_ROOT
  torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT \
    -m parser.joint_train --config $CFG0
"
echo JOINT_HF_DONE
