#!/bin/bash
# Run W&B optimizer sweep on RunPod.
#
# Usage:
#   # Create sweep and start agent (first time):
#   ./runpod/run_optimizer_sweep.sh
#
#   # Join existing sweep (pass sweep ID from wandb output):
#   ./runpod/run_optimizer_sweep.sh <sweep_id>
#
#   # Run in background (survives SSH disconnect):
#   nohup ./runpod/run_optimizer_sweep.sh [sweep_id] > sweep.log 2>&1 &

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
WANDB_ENTITY="${WANDB_ENTITY:-jbmopper-0}"
WANDB_PROJECT="cs336-optimizer-sweep"
SWEEP_CONFIG="cs336_basics/optimizer_sweep_config.yaml"

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ ! -f "${DATA_DIR}/tinystories_train_fixed.npy" ]]; then
    if [[ -f "tokenized/tinystories_train_fixed.npy" ]]; then
        DATA_DIR="tokenized"
    else
        echo "ERROR: Tokenized data not found at ${DATA_DIR} or ./tokenized/"
        echo "Run ./runpod/sync_data.sh first"
        exit 1
    fi
fi
export DATA_DIR

# Symlink for any code that expects ./tokenized/
if [[ ! -d "tokenized" ]] && [[ -d "${DATA_DIR}" ]] && [[ "${DATA_DIR}" != "tokenized" ]]; then
    ln -sf "${DATA_DIR}" tokenized
fi

# Verify wandb is authenticated
if ! uv run python -c "import wandb; wandb.Api()" 2>/dev/null; then
    echo "ERROR: wandb not authenticated. Set WANDB_API_KEY or run 'wandb login'."
    exit 1
fi

# GPU info
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected')
" 2>/dev/null || true

echo ""
echo "============================================"
echo "CS336 Optimizer Sweep"
echo "============================================"
echo "Entity:  ${WANDB_ENTITY}"
echo "Project: ${WANDB_PROJECT}"
echo "Data:    ${DATA_DIR}"
echo ""

# ── Create or join sweep ─────────────────────────────────────────────────────

SWEEP_ID="${1:-}"

if [[ -z "${SWEEP_ID}" ]]; then
    echo "Creating new sweep from ${SWEEP_CONFIG}..."
    SWEEP_ID=$(uv run wandb sweep "${SWEEP_CONFIG}" 2>&1 | grep -oP '[\w-]+/[\w-]+/[\w]+$' || true)

    if [[ -z "${SWEEP_ID}" ]]; then
        echo "Could not parse sweep ID. Creating sweep manually..."
        uv run wandb sweep "${SWEEP_CONFIG}"
        echo ""
        echo "Copy the sweep ID from above and re-run:"
        echo "  ./runpod/run_optimizer_sweep.sh <sweep_id>"
        exit 0
    fi
    echo "Created sweep: ${SWEEP_ID}"
else
    echo "Joining existing sweep: ${SWEEP_ID}"
fi

echo ""
echo "Starting sweep agent..."
echo "  (Ctrl+C or kill to stop; the sweep persists on wandb.ai)"
echo ""

uv run wandb agent "${SWEEP_ID}"
