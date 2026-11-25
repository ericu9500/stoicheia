#!/bin/bash
#SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
#SBATCH -p gpu   # EDIT: your GPU partition name
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH --job-name=ithaca-gpu
#SBATCH -o logs/ithaca-%j.out
#SBATCH -e logs/ithaca-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Run the DeepMind Ithaca baseline on a GH200 (JAX aarch64 CUDA from the local
# wheelhouse — compute nodes have no internet). First run builds the venv.
# Usage: sbatch scripts/slurm/insc_ithaca_gpu.sbatch <split val|test> <n_per_length> <out.json> <exclude.json>
set -euo pipefail
SPLIT="${1:?split}"; N="${2:?n}"; OUT="${3:?out}"; EXCL="${4:?exclude}"; LENGTHS="${5:-1,2,3,4,5,6,7,8,9,10}"; SAMPLES="${6:-}"; SHARD="${7:-}"
INS_ROOT=$STOICHEIA_ROOT
STOICHEIA_ROOT=$STOICHEIA_ROOT
source $STOICHEIA_ROOT/env.sh >/dev/null 2>&1 || true

apptainer exec --nv $APPTAINER_BINDS $SIF bash -c "
  set -e
  VENV=$INS_ROOT/.venv-ithaca-arm
  if [ ! -f \$VENV/bin/python ]; then
    python3 -m venv \$VENV
    \$VENV/bin/pip install -q --no-index --find-links $INS_ROOT/wheels_arm_ithaca \
      'jax[cuda12]==0.4.35' dm-haiku flax chex ml-collections absl-py tqdm numpy
    echo 'venv built'
  fi
  export PYTHONPATH=$INS_ROOT/ithaca_upstream:$STOICHEIA_ROOT
  # jax 0.4.35's nvcc namespace-package probe crashes; point it at the container CUDA
  export CUDA_ROOT=/usr/local/cuda CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
  \$VENV/bin/python -c 'import jax; print(\"jax devices:\", jax.devices())'
  cd $INS_ROOT
  \$VENV/bin/python insc/eval/ithaca_baseline.py --split $SPLIT --n $N --workers 1 \
    --lengths $LENGTHS --exclude $EXCL --out $OUT ${SAMPLES:+--samples $SAMPLES} ${SHARD:+--shard $SHARD}
"
echo ITHACA_GPU_DONE
