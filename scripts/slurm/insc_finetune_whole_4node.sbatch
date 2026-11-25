#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 4
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 128
#SBATCH -t 10:00:00
#SBATCH --job-name=insc-ftw2
#SBATCH -o logs/ftwhole-%j.out
#SBATCH -e logs/ftwhole-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# 4-node (16-GPU) whole-document documentary finetune (torchrun) -- finetune_whole.py
# already has full DDP support (ddp_setup()/DDP wrap, the same pattern as pretraining), this
# just launches it under torchrun instead of plain python. Use attn=sdpa in the configs
# (the compiled flex_attention block-sparse kernel can fall back to a memory-exploding
# dense computation at these sequence lengths).
# Usage: sbatch scripts/slurm/insc_finetune_whole_4node.sbatch <config>
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/slurm/insc_finetune_whole_4node.sbatch <config>}"
INS_ROOT=$STOICHEIA_ROOT

source "$INS_ROOT/env.sh"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "config=$CONFIG nnodes=$SLURM_NNODES ngpu=$NGPU $(date)"

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
      insc/train/finetune_whole.py --config $INS_ROOT/$CONFIG
  "
echo FT_JOB_END
