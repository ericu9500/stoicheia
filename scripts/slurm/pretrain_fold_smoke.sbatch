#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 00:25:00
#SBATCH --job-name=gcb-fold-smoke
#SBATCH -o logs/fold-smoke-%j.out
#SBATCH -e logs/fold-smoke-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Fold-data smoke: 150 steps on 2 nodes against fold_0's shards, BEFORE launching the
# 10-chain fleet. Usage: sbatch scripts/slurm/pretrain_fold_smoke.sbatch
set -euo pipefail
STOICHEIA_ROOT=$STOICHEIA_ROOT
STOICHEIA_DATA=${STOICHEIA_DATA:?set STOICHEIA_DATA (source env.sh in your login shell before submitting)}
mkdir -p "$STOICHEIA_ROOT/logs"

# plain system python3 (aarch64-native) — see smoke.sbatch for why not the x86 venv here
python3 -c "
import json
c = json.load(open('$STOICHEIA_ROOT/configs/pretrain/folds/fold_0.json'))
c['name'] = 'fold0_smoke'
c['total_steps'] = 150
c['ckpt_every'] = 100
c['log_every'] = 5
c['eval_n'] = 64
c['out_dir'] = '$STOICHEIA_DATA/runs/fold0_smoke'
json.dump(c, open('$STOICHEIA_ROOT/configs/pretrain/_fold_smoke.json', 'w'), indent=2)
"

# smoke is a stateless correctness check — never resume a stale run
rm -rf "$STOICHEIA_DATA/runs/fold0_smoke"

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))  # avoid fixed-port collisions
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $STOICHEIA_ROOT/env.sh
  export STOICHEIA_DATA_ROOT=$STOICHEIA_DATA/folds/fold_0
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
      -m train.train --config $STOICHEIA_ROOT/configs/pretrain/_fold_smoke.json
  '
"
echo FOLD_SMOKE_DONE
