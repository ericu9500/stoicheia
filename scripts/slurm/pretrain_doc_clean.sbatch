#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 32              # 32 nodes x 4 GH200 = 128 GPUs (same recipe as final.sbatch)
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 1
#SBATCH -c 256
#SBATCH -t 08:00:00
#SBATCH --job-name=gcb-doc
#SBATCH -o logs/doc-%j.out
#SBATCH -e logs/doc-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# The documentary-clean replica: flagship recipe on the corpus with all papyri
# (ddbdp/dclp) dropped and every sentence echoing any PHI inscription or papyrus
# excised (see data_split/pipeline/{04b,05b,06b}*.py). Clean for reconstruction on
# ANY edited inscription or documentary papyrus. Usage: sbatch scripts/slurm/pretrain_doc_clean.sbatch
set -euo pipefail
source $STOICHEIA_ROOT/env.sh
mkdir -p "$STOICHEIA_ROOT/logs"

CONFIG=$STOICHEIA_ROOT/configs/pretrain/folds/doc_clean.json
DOC_ROOT=$STOICHEIA_DATA/folds/doc_clean
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
[ -f "$DOC_ROOT/shards/v1_punct/index.parquet" ] || { echo "missing shards at $DOC_ROOT"; exit 1; }

MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_ADDR=$MASTER MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))  # avoid fixed-port collisions
NGPU=${SLURM_GPUS_PER_NODE:-4}
echo "doc_clean nodes=$SLURM_NNODES master=$MASTER gpus/node=$NGPU config=$CONFIG $(date)"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -lc "
  source $STOICHEIA_ROOT/env.sh
  export STOICHEIA_DATA_ROOT=$DOC_ROOT
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
echo "DOC_CLEAN JOB FINISHED (or checkpointed out at wall-time limit — chain resumes)"
