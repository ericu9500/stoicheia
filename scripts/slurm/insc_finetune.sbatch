#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 4
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 06:00:00
#SBATCH --job-name=insc-ft
#SBATCH -o logs/ft-%j.out
#SBATCH -e logs/ft-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Usage: sbatch scripts/slurm/insc_finetune.sbatch [config]
set -euo pipefail
CONFIG="${1:-configs/insc/finetune.json}"
INS_ROOT=$STOICHEIA_ROOT

# Sourced ONCE here, at the top-level sbatch shell -- not inside the per-node srun'd shell
# below. Re-sourcing env.sh inside a remote `bash -lc` on each freshly srun'd node depends on
# that login shell's own startup chain completing cleanly; on this cluster it doesn't (see
# this launcher's history) and $SIF/$APPTAINER_BINDS silently never get set, so
# apptainer misparses the following "bash" as the image path and dies in ~seconds. Resolving
# them here instead means every node gets the exact same, already-correct values baked into
# its own command line, no remote re-sourcing required.
source "$INS_ROOT/env.sh"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER config=$CONFIG $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=16
    cd $INS_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      insc/train/finetune.py --config $INS_ROOT/$CONFIG
  "
echo FT_JOB_END
