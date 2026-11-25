#!/bin/bash
# Build the self-contained CPU venv (data pipeline / unit tests) at $STOICHEIA_ROOT/.venv-cpu.
# Requires `uv` on PATH. GPU-side python comes from the training container instead.
# NOTE: this script lives at scripts/pretrain/setup_venv.sh (two levels below repo root).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STOICHEIA_DATA=${STOICHEIA_DATA:?set STOICHEIA_DATA/STOICHEIA_DATA (source env.sh first)}

export UV_PYTHON_INSTALL_DIR=$STOICHEIA_DATA/cache/uv-pythons   # keep interpreters off $HOME
uv venv --clear --python 3.12 "$ROOT/.venv-cpu"
# unsafe-best-match: the pytorch extra index carries old numpy versions; let pypi win there
uv pip install --python "$ROOT/.venv-cpu/bin/python" --index-strategy unsafe-best-match \
    -r "$ROOT/requirements-cpu.txt"
"$ROOT/.venv-cpu/bin/python" -c "import torch, numpy, pyarrow, pytest, huggingface_hub; print('venv OK')"
