#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH --job-name=insc-leakfold
#SBATCH -o logs/leakfold-%j.out
#SBATCH -e logs/leakfold-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was invoked from,
# immune to that copy (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Per-fold-digit contamination scan (iphi + papyri, val + test) against the ACTUAL
# pretraining backbone every restoration fold fine-tunes from (stoicheia_doc_clean) -- not the
# flagship v1_punct/bronze_punct corpus the un-parameterized leak_scan.py defaults point
# at. All 10 restoration folds share this ONE un-rotated backbone (only the fine-tuning
# stage rotates by digit), so contamination for fold t{X}v{Y} is whatever of digit X's
# segments leaked into doc_clean's OWN pretraining shards -- checked per-digit here since
# only fold0's exclusion lists existed before this (built for the original, pre-rotation
# single-fold setup).
# Usage: sbatch scripts/slurm/insc_leak_scan_fold.sbatch <test_digit>
set -euo pipefail
TEST_DIGIT="${1:?usage: sbatch scripts/slurm/insc_leak_scan_fold.sbatch <test_digit>}"
INS_ROOT=$STOICHEIA_ROOT

source "$INS_ROOT/env.sh"

DOC_CLEAN_PLANES="$STOICHEIA_DATA/folds/doc_clean/shards/v1_punct/chars.bin,$STOICHEIA_DATA/folds/doc_clean/shards/bronze_punct/chars.bin"

echo "test_digit=$TEST_DIGIT planes=$DOC_CLEAN_PLANES $(date)"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export INSC_TEST_DIGIT=$TEST_DIGIT
    cd $INS_ROOT
    python insc/eval/leak_scan.py     --split test --workers 8 --planes $DOC_CLEAN_PLANES --out \$INS_DATA/contaminated_test_fold${TEST_DIGIT}_docclean.json
    python insc/eval/pap_leak_scan.py --split test --workers 8 --planes $DOC_CLEAN_PLANES --out \$INS_DATA/contaminated_pap_test_fold${TEST_DIGIT}_docclean.json
  "
echo LEAK_SCAN_FOLD_DONE
