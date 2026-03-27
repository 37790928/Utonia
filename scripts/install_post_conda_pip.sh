#!/usr/bin/env bash
# Run after: conda env create -f environment.yml && conda activate utonia
# Conda already installed PyTorch. These packages need that torch at pip build/install time.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  echo "Activate the utonia conda env first: conda activate utonia" >&2
  exit 1
fi

PYG_WHEELS="https://data.pyg.org/whl/torch-2.5.0+cu124.html"

echo "Installing torch-scatter (needs conda PyTorch; --no-build-isolation avoids pip's empty build env)..."
python -m pip install --no-build-isolation "torch-scatter" -f "${PYG_WHEELS}"

echo "Installing FlashAttention..."
python -m pip install --no-build-isolation "flash-attn" \
  || python -m pip install --no-build-isolation "git+https://github.com/Dao-AILab/flash-attention.git"

echo "Done."
