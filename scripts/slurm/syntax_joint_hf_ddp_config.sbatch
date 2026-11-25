#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 8                 # 8 nodes x 4 GH200 = 32 GPUs
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 04:00:00
#SBATCH --job-name=joint-hf-ddp
#SBATCH -o logs/jointhfddp-%j.out
#SBATCH -e logs/jointhfddp-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Same-recipe cross-encoder ablation, full 32-GPU multi-node DDP -- generic version that
# takes any config filename (not just fold0/seed0). Usage:
#   sbatch scripts/slurm/syntax_joint_hf_ddp_config.sbatch <config.json>
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/joint_hf_ddp_config.sbatch <config.json>}"
SYN_ROOT=$STOICHEIA_ROOT
source "$SYN_ROOT/env.sh"
CONFIG="$SYN_ROOT/configs/syntax/$CONFIG"
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
mkdir -p "$SYN_ROOT/logs"
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU eff_gpus=$((SLURM_NNODES*NGPU)) config=$CONFIG $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $SYN_ROOT/env.sh
  apptainer exec --nv \$APPTAINER_BINDS \$SIF bash -lc '
    set -e
    export PYTHONPATH=$SYN_ROOT:$TAGGER_ROOT:$STOICHEIA_ROOT
    export HF_HOME=$HF_HOME
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
    cd $SYN_ROOT
    torchrun \
      --nnodes=$SLURM_NNODES --nproc_per_node=$NGPU \
      --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      --node_rank=\$SLURM_PROCID \
      -m parser.joint_train --config $CONFIG
  '
"
echo "JOINT_HF_DDP_DONE $(date)"
