# Stoicheia environment. Source this before running anything.
#   export STOICHEIA_DATA=/path/to/writable/data+checkpoints
#   source env.sh
# Everything else derives from those two roots.

# This repository's checkout.
export STOICHEIA_ROOT="${STOICHEIA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# Writable data root: corpora, shards, checkpoints, eval outputs.
export STOICHEIA_DATA="${STOICHEIA_DATA:?set STOICHEIA_DATA to a writable data+checkpoint root}"

# Per-task subtrees under the data root. Configs and scripts reference these by name.
export INS_DATA="$STOICHEIA_DATA/insc_data"                            # documentary corpora + runs
export SYN_DATA="$STOICHEIA_DATA/parser_data"                          # treebank shards + parser runs
export METER_DATA="$STOICHEIA_DATA/meter"                              # encoded meter training data
export AGD_DATA="${AGD_DATA:-$STOICHEIA_DATA/agd}"                     # normalized source corpora
# Checkout of the released meter-silver dataset (verse lines, scanner corpus, Norma).
# Only needed to retrain the meter model or to run its data-dependent tests.
export MACRONIZER_SRC="${MACRONIZER_SRC:-$STOICHEIA_DATA/macron_data}"

# Sub-packages resolve their code root through these; all are this checkout.
export INS_ROOT="$STOICHEIA_ROOT"
export SYN_ROOT="$STOICHEIA_ROOT"
export TAGGER_ROOT="$STOICHEIA_ROOT"
export METER_ROOT="$STOICHEIA_ROOT"

# Pretrained torso used by the downstream configs that do not name a checkpoint inline.
export STOICHEIA_CKPT="${STOICHEIA_CKPT:-$STOICHEIA_DATA/runs/stoicheia_doc_clean/best.pt}"

export PYTHONPATH="$STOICHEIA_ROOT:${PYTHONPATH:-}"

# Caches inside the data root, so nothing lands in a read-only or shared HOME.
export HF_HOME="$STOICHEIA_DATA/cache/huggingface"
export XDG_CACHE_HOME="$STOICHEIA_DATA/cache/xdg"
export TRITON_CACHE_DIR="$STOICHEIA_DATA/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$STOICHEIA_DATA/cache/inductor"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null

# Optional containerized execution (the SLURM templates use it; leave unset to run in a venv).
#   export STOICHEIA_SIF=/path/to/container.sif
#   export APPTAINER_BINDS="-B $STOICHEIA_DATA"
export SIF="${STOICHEIA_SIF:-}"
