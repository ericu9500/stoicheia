#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 00:25:00
#SBATCH --job-name=gcb-smoke
#SBATCH -o logs/smoke-%j.out
#SBATCH -e logs/smoke-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Quick correctness + memory + throughput check BEFORE committing the full fleet.
# Usage: sbatch scripts/slurm/pretrain_smoke.sbatch
set -euo pipefail
STOICHEIA_ROOT=$STOICHEIA_ROOT
STOICHEIA_DATA=${STOICHEIA_DATA:?set STOICHEIA_DATA (source env.sh in your login shell before submitting)}
mkdir -p "$STOICHEIA_ROOT/logs"

# plain system python3 (aarch64-native on this node) — NOT the x86 venv from env.sh, which
# would shadow python3 with a binary that can't execute here (env.sh is sourced later, inside
# the container, where it's the right architecture).
python3 -c "
import json
c = json.load(open('$STOICHEIA_ROOT/configs/pretrain/stoicheia.json'))
c['name'] = 'smoke'
c['total_steps'] = 150
c['ckpt_every'] = 100
c['log_every'] = 5
c['eval_n'] = 64
c['out_dir'] = '$STOICHEIA_DATA/runs/pretrain_smoke'
json.dump(c, open('$STOICHEIA_ROOT/configs/pretrain/_smoke.json', 'w'), indent=2)
"

# smoke is a stateless correctness check — never resume a stale run
rm -rf "$STOICHEIA_DATA/runs/pretrain_smoke"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))  # avoid fixed-port collisions
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $STOICHEIA_ROOT/env.sh
  source $STOICHEIA_ROOT/scripts/pretrain/stage_shards.sh
  apptainer exec --nv \$APPTAINER_BINDS \$STAGE_BIND \$SIF bash -lc '
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $STOICHEIA_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      -m train.train --config $STOICHEIA_ROOT/configs/pretrain/_smoke.json
  '
"
echo SMOKE_DONE
