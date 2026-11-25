#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH --job-name=joint-hf-eval
#SBATCH -o logs/jointhfeval-%j.out
#SBATCH -e logs/jointhfeval-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Held-out TEST split eval for a finished HF-backbone joint run.
# Usage: sbatch scripts/slurm/syntax_joint_hf_eval.sbatch <encoder_tag>
set -euo pipefail
TAG="${1:?usage: sbatch scripts/joint_hf_eval.sbatch <encoder_tag>}"
SYN_ROOT=$STOICHEIA_ROOT
STOICHEIA_ROOT=$STOICHEIA_ROOT
source $STOICHEIA_ROOT/env.sh
source $SYN_ROOT/env.sh

apptainer exec --nv $APPTAINER_BINDS $SIF bash -c "
  set -e
  export PYTHONPATH=$SYN_ROOT:$TAGGER_ROOT:$STOICHEIA_ROOT
  export HF_HOME=$HF_HOME
  cd $SYN_ROOT
  python -m parser.joint_evaluate --run \$SYN_DATA/runs/joint_${TAG}_f0_s0 --split test
"
echo JOINT_HF_EVAL_DONE
