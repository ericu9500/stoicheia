#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node 1
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH --job-name=insc-evalfold
#SBATCH -o logs/evalfold-%j.out
#SBATCH -e logs/evalfold-%j.out
# SLURM copies the submitted script into a per-job spool dir before running it on this
# cluster, so locating the repo via ${BASH_SOURCE[0]} resolves to that spool path, not
# the real one ("/var/lib/slurm/..." errors downstream) -- a real failure mode hit
# repeatedly in practice. $SLURM_SUBMIT_DIR is set by sbatch to the directory it was
# invoked from, immune to that copy, and matches this repo's own submit-from-root
# convention (see scripts/slurm/README.md point 6).
STOICHEIA_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
export STOICHEIA_ROOT
# Per-fold CER/top1/top20 on EACH fold's own held-out test digit, evaluated separately on
# the inscriptions (iphi) and papyri test sets -- the un-conditioned baseline the new
# metadata-primed models need to beat. Unlike insc_eval.sbatch (fixed digit 3/4), this
# parameterizes test/val digit per fold so digit-3 records aren't wrongly treated as
# held-out for folds whose OWN held-out digit is something else.
# Usage: sbatch scripts/slurm/insc_eval_fold.sbatch <ckpt> <test_digit> <val_digit> <out_prefix> [n_per_length]
# Contamination exclusion lists (per-digit, against the shared doc_clean pretraining
# backbone -- see insc_leak_scan_fold.sbatch) are picked up automatically from
# $INS_DATA/contaminated_test_fold<X>_docclean.json / contaminated_pap_test_fold<X>_docclean.json
# if present; a missing file is a hard error, not a silent skip, so a fold can't
# accidentally get scored uncontaminated-vs-contaminated against the others.
set -euo pipefail
CKPT="${1:?usage: sbatch scripts/slurm/insc_eval_fold.sbatch <ckpt> <test_digit> <val_digit> <out_prefix> [n]}"
TEST_DIGIT="${2:?test_digit required}"
VAL_DIGIT="${3:?val_digit required}"
OUT_PREFIX="${4:?out_prefix required}"
N="${5:-200}"
INS_ROOT=$STOICHEIA_ROOT

source "$INS_ROOT/env.sh"

EXCL_IPHI="$INS_DATA/contaminated_test_fold${TEST_DIGIT}_docclean.json"
EXCL_PAP="$INS_DATA/contaminated_pap_test_fold${TEST_DIGIT}_docclean.json"
[ -f "$EXCL_IPHI" ] || { echo "MISSING $EXCL_IPHI -- run insc_leak_scan_fold.sbatch $TEST_DIGIT first" >&2; exit 1; }
[ -f "$EXCL_PAP" ]  || { echo "MISSING $EXCL_PAP -- run insc_leak_scan_fold.sbatch $TEST_DIGIT first" >&2; exit 1; }

echo "ckpt=$CKPT test_digit=$TEST_DIGIT val_digit=$VAL_DIGIT excl_iphi=$EXCL_IPHI excl_pap=$EXCL_PAP $(date)"

srun apptainer exec --nv $APPTAINER_BINDS $SIF bash -lc "
    set -e
    export PYTHONPATH=$STOICHEIA_ROOT
    export INSC_TEST_DIGIT=$TEST_DIGIT INSC_VAL_DIGIT=$VAL_DIGIT
    cd $INS_ROOT
    python insc/eval/restore.py --ckpt $CKPT --split test --domain iphi   --n $N --exclude $EXCL_IPHI --out ${OUT_PREFIX}_iphi.json
    python insc/eval/restore.py --ckpt $CKPT --split test --domain papyri --n $N --exclude $EXCL_PAP  --out ${OUT_PREFIX}_papyri.json
  "
echo EVAL_FOLD_JOB_END
