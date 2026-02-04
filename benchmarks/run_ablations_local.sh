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

# W&B configuration (can override via env vars)
WANDB_PROJECT="${WANDB_PROJECT:-cs336-a1}"
WANDB_ENTITY="${WANDB_ENTITY:-jbmopper-0}"

# Training hyperparameters
NUM_ITERS=5000
WARMUP_ITERS=500     # 10% warmup (common heuristic)
COSINE_ITERS=5000    # Cosine decay over full training length

COMMON_ARGS="--data-dir ${DATA_DIR} --precision ${PRECISION} --num-iters ${NUM_ITERS} --warmup-iters ${WARMUP_ITERS} --cosine-iters ${COSINE_ITERS} --eval-every 50 --save-best --save-final --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"

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

# Use batch_size=32 for all ablations (assignment spec for local/experiment runs)
BATCH_SIZE=32
echo "Using batch_size=32 for all ablations"

# Assignment model config (17M params)
MODEL_ARGS="--d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --batch-size ${BATCH_SIZE}"

echo ""
echo "=== Assignment Architecture Ablations ==="
echo "Base config: d_model=512, num_layers=4, num_heads=16, d_ff=1344"
echo ""

# 0. Baseline (Pre-norm + RoPE + SwiGLU)
run_exp "baseline" \
    $MODEL_ARGS

# 1. Layer Norm Ablation - Remove RMSNorm entirely
run_exp "no_norm" \
    $MODEL_ARGS --norm-mode none

# 2. Post-norm Ablation - Post-norm instead of pre-norm
run_exp "post_norm" \
    $MODEL_ARGS --norm-mode post

# 3. Position Embedding Ablation - NoPE (no position embeddings)
run_exp "nope" \
    $MODEL_ARGS --no-rope

# 4. FFN Ablation - SiLU (non-gated) instead of SwiGLU (gated)
# Note: SiLU uses d_ff=4*d_model=2048 to match parameter count
run_exp "silu" \
    $MODEL_ARGS --ffn-type silu --ffn-hidden-dim 2048

echo ""
echo "All ablations complete!"
