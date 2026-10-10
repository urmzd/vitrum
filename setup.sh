#!/usr/bin/env bash
# Setup vitrum on a fresh machine (e.g. RunPod)
# Usage: bash setup.sh
set -euo pipefail

cd /workspace

# Clone if needed
[ -d vitrum ] || git clone https://github.com/urmzd/vitrum.git
cd vitrum

# Install workspace members into system Python (torchrun uses system Python,
# not a venv). All members are passed explicitly so the local editables
# resolve each other instead of PyPI.
uv pip install --system \
    -e libs/core -e libs/architectures -e libs/data -e libs/training -e apps/cli

# Download data
vitrum data download --variant sp1024

echo "Setup complete. Run training with:"
echo "  cd /workspace/vitrum"
echo "  MODEL_VERSION=v8_lowrank_vv torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) -m vitrum.training"
