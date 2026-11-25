#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 32              # 32 nodes x 4 GH200 = 128 GPUs (drop to 16 for 64 GPUs)
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 08:00:00
#SBATCH --job-name=gcb-final
#SBATCH -o logs/final-%j.out
#SBATCH -e logs/final-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Usage: sbatch scripts/slurm/pretrain_final.sbatch configs/pretrain/stoicheia.json
# Multi-node torchrun with c10d rendezvous on the batch master. Safe to submit as a CHAIN
# of jobs (sbatch --dependency=afterany:<prev_jobid> ...) — the trainer checkpoints every
# ckpt_every steps and auto-resumes from out_dir/last.pt, so an 8h wall-time slot per job
# schedules far more easily than one long monolithic request, and a crash only loses at
# most one checkpoint interval.
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/final.sbatch <config.json>}"
source $STOICHEIA_ROOT/env.sh
mkdir -p "$STOICHEIA_ROOT/logs"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))  # avoid fixed-port collisions
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU config=$CONFIG $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $STOICHEIA_ROOT/env.sh
  source $STOICHEIA_ROOT/scripts/pretrain/stage_shards.sh
  apptainer exec --nv \$APPTAINER_BINDS \$STAGE_BIND \$SIF bash -lc '
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
    cd $STOICHEIA_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      -m train.train --config $CONFIG
  '
"
echo 'FINAL RUN FINISHED (or checkpointed out at wall-time limit — resume by resubmitting)'
