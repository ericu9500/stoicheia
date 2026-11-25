#!/bin/bash
# Build the GPU venv. Run ON a GPU node (never build
# compiled wheels for another architecture from a login node of a different arch).
#   bash scripts/setup_gpu_venv.sh /path/to/venv
# Tested: NVIDIA GH200 (aarch64), driver 580 / CUDA 13, torch 2.12.1+cu130. Other CUDA versions: change the
# extra index in requirements-gpu.txt (e.g. cu128) and the torch local version accordingly.
set -euo pipefail
VENV=${1:?usage: setup_gpu_venv.sh <venv dir>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r "$ROOT/requirements-gpu.txt"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"
(cd "$ROOT" && "$VENV/bin/python" -m pytest tests -q)
