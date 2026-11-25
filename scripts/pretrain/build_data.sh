#!/bin/bash
# Reproducible data build: HuggingFace -> raw parquet/jsonl -> memmap shards.
# Run once on a CPU (x86) node before training; training only reads the shards.
#   scripts/pretrain/build_data.sh [workers]
set -euo pipefail
WORKERS=${1:-16}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/env.sh"
PY="${PYTHON:-python3}"
RAW=$STOICHEIA_DATA/raw

$PY "$ROOT/data/fetch_hf.py" --out "$RAW"
$PY "$ROOT/data/build_shards.py" --raw "$RAW/AncientGreek/data" \
    --out "$STOICHEIA_DATA/shards/v1_punct" --workers "$WORKERS"
$PY "$ROOT/data/build_bronze.py" --jsonl "$RAW/SyntheticAncientGreek-CorpusCorporum/bronze.jsonl" \
    --out "$STOICHEIA_DATA/shards/bronze_punct" --workers "$WORKERS"
cp "$RAW/provenance.json" "$STOICHEIA_DATA/shards/provenance.json"
echo "DATA BUILD DONE: $STOICHEIA_DATA/shards/{v1_punct,bronze_punct}"
