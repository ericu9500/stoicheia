#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH --job-name=syn-jeval
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
#   sbatch scripts/slurm/syntax_joint_eval.sbatch $SYN_DATA/runs/joint_docclean_f3_s0
set -euo pipefail
SYN_ROOT=$STOICHEIA_ROOT
source "$SYN_ROOT/env.sh"
RUN="${1:?usage: sbatch scripts/slurm/syntax_joint_eval.sbatch <run_dir> [greedy|mst]}"
export DECODE="${2:-greedy}"
srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
  export PYTHONPATH=$SYN_ROOT:$TAGGER_ROOT:$STOICHEIA_ROOT
  export STOICHEIA_CKPT=$STOICHEIA_CKPT
  cd $SYN_ROOT
  python -u -m parser.joint_evaluate --run $RUN --split test --decode ${DECODE:-greedy}
"
echo EVAL_DONE
