#!/bin/bash
# Run local ablations sequentially.
# Usage: ./benchmarks/run_ablations_local.sh [gpu_id]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/tokenized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/ablations}"

# Common settings
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: Tokenized data not found at ${DATA_DIR}"
    echo "Set DATA_DIR or sync tokenized data."
    exit 1
fi

# Auto-detect precision based on device
# MPS (Mac) doesn't support bf16, CUDA supports bf16
PRECISION="${PRECISION:-fp32}"
if command -v nvidia-smi &> /dev/null; then
    PRECISION="bf16"
    echo "Detected CUDA GPU - using bf16 precision"
else
    echo "No CUDA GPU detected - using fp32 precision (MPS compatible)"
fi

COMMON_ARGS="--data-dir ${DATA_DIR} --precision ${PRECISION} --num-iters 5000 --eval-every 500"

echo "Starting local ablations on GPU $GPU_ID..."
echo "Common args: $COMMON_ARGS"

# Function to run an experiment
run_exp() {
    NAME="$1"
    shift
    echo ""
    echo "=== Running Experiment: $NAME ==="
    uv run python -m cs336_basics.train \
        --run-name "ablation_${NAME}" \
        --checkpoint-dir "${CHECKPOINT_DIR}/${NAME}" \
        $COMMON_ARGS \
        "$@"
}

# --- Define Experiments Below ---

# Auto-detect batch size based on device
# CUDA GPUs can handle batch_size=64, MPS/CPU should use 32
BATCH_SIZE="${BATCH_SIZE:-64}"
if ! command -v nvidia-smi &> /dev/null; then
    BATCH_SIZE=32
    echo "No CUDA GPU detected - using batch_size=32 (MPS/CPU compatible)"
else
    echo "CUDA GPU detected - using batch_size=64"
fi

# 1. Baseline (Assignment Config)
run_exp "baseline" \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --batch-size ${BATCH_SIZE}

# 2. Wide (More heads/width, fewer layers)
run_exp "wide" \
    --d-model 768 --num-layers 2 --num-heads 12 --d-ff 2048 --batch-size ${BATCH_SIZE}

# 3. Deep (More layers, narrower)
run_exp "deep" \
    --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1024 --batch-size ${BATCH_SIZE}

# 4. Wider FFN (Test FFN scaling)
run_exp "wide_ffn" \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 2048 --batch-size ${BATCH_SIZE}

echo ""
echo "All ablations complete!"
