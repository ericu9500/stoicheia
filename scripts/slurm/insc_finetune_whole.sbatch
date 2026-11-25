#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH --job-name=insc-ftwhole
#SBATCH -o logs/ftwhole-%j.out
#SBATCH -e logs/ftwhole-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Single-GPU whole-document (lacuna-preserving) documentary restoration finetune. Unlike
# the 4-node launcher (4-node x 4-GPU, shard-based, mixes in GB-scale GCB literary
# replay), this is in-memory over just the whole PHI+papyri corpus (~60M characters) --
# far smaller, so one GPU is appropriate. Each fold trains from the SAME stoicheia_doc_clean
# backbone but excludes its own test+val digit's documents entirely from training.
# Usage: sbatch scripts/slurm/insc_finetune_whole.sbatch <config>
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/slurm/insc_finetune_whole.sbatch <config>}"
INS_ROOT=$STOICHEIA_ROOT

source "$INS_ROOT/env.sh"

echo "config=$CONFIG $(date)"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $INS_ROOT
    python insc/train/finetune_whole.py --config $INS_ROOT/$CONFIG
  "
echo FT_JOB_END
