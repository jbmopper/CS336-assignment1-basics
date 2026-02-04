#!/usr/bin/env bash
# Prepare environment, sync data, and run ablation experiments with background S3 sync.
#
# Usage:
#   ./runpod/prep_run_ablations.sh [s3_output_url] [gpu_id]
#
# Arguments:
#   s3_output_url : S3 path to save results (default: s3://.../ablations_results)
#   gpu_id        : GPU device ID to use (default: 0)
#
# Example:
#   ./runpod/prep_run_ablations.sh s3://my-bucket/ablations 0

set -euo pipefail

# --- Configuration ---
S3_DEFAULT_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
S3_OUTPUT="${1:-${S3_DEFAULT_BUCKET}/ablations_results/${RUN_TS}}"
GPU_ID="${2:-0}"

# Note: W&B is enabled by default in train.py
# Assignment ablations test architectural components:
#   - ablation_baseline: Pre-norm + RoPE + SwiGLU (default)
#   - ablation_no_norm: Remove all layer normalization
#   - ablation_post_norm: Post-norm instead of pre-norm
#   - ablation_nope: No position embeddings (NoPE)
#   - ablation_silu: SiLU (non-gated) instead of SwiGLU (gated)
# All logged to project: cs336-a1

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_TOKENIZED="${S3_TOKENIZED:-${S3_DEFAULT_BUCKET}/tokenized/}"
DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
TOKENIZED_SYMLINK="${REPO_ROOT}/tokenized"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/ablations}"

# --- 1. Data Setup ---
echo "============================================"
echo "Step 1: Syncing Data"
echo "============================================"
echo "Downloading tokenized data from: ${S3_TOKENIZED}"
echo "Destination: ${DATA_DIR}"

mkdir -p "${DATA_DIR}"
aws s3 sync "${S3_TOKENIZED}" "${DATA_DIR}" --no-progress

# Create symlink for local scripts
if [[ ! -e "${TOKENIZED_SYMLINK}" ]]; then
    ln -sf "${DATA_DIR}" "${TOKENIZED_SYMLINK}"
    echo "Created symlink: ${TOKENIZED_SYMLINK} -> ${DATA_DIR}"
fi

# Ensure expected filenames exist (legacy support)
if [[ -f "${TOKENIZED_SYMLINK}/tinystories_train.npy" ]]; then
    ln -sf "tinystories_train.npy" "${TOKENIZED_SYMLINK}/tinystories_train_fixed.npy" 2>/dev/null || true
fi
if [[ -f "${TOKENIZED_SYMLINK}/tinystories_valid.npy" ]]; then
    ln -sf "tinystories_valid.npy" "${TOKENIZED_SYMLINK}/tinystories_valid_fixed.npy" 2>/dev/null || true
fi

# --- 2. Checkpoint Directory Setup ---
echo ""
echo "============================================"
echo "Step 2: Setting up Checkpoint Directory"
echo "============================================"
echo "Checkpoints will be saved to: ${CHECKPOINT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# --- 3. Output Sync Setup ---
echo ""
echo "============================================"
echo "Step 3: Starting Background Sync"
echo "============================================"
echo "Local Checkpoints: ${CHECKPOINT_DIR}"
echo "Remote S3:         ${S3_OUTPUT}"

# Start sync script in background (sync every 60s for checkpoints)
nohup bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" 60 \
    > /tmp/s3sync_ablations.log 2>&1 &
SYNC_PID=$!
echo "Background sync started (PID: ${SYNC_PID})"
echo "Sync log: /tmp/s3sync_ablations.log"

# Register cleanup trap to ensure final sync
cleanup() {
    echo ""
    echo "============================================"
    echo "Stopping background sync..."
    kill "${SYNC_PID}" 2>/dev/null || true
    
    echo "Performing FINAL sync..."
    bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" 0
    echo "Done."
}
trap cleanup EXIT

# --- 4. Environment Check ---
echo ""
echo "============================================"
echo "Step 4: Environment Check"
echo "============================================"
echo "GPU_ID: ${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Check for CUDA and bf16 support
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected - bf16 will be used automatically"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${GPU_ID}" || true
    export PRECISION="${PRECISION:-bf16}"
else
    echo "WARNING: CUDA not detected - falling back to fp32"
    export PRECISION="${PRECISION:-fp32}"
fi

# --- 5. Run Ablations ---
echo ""
echo "============================================"
echo "Step 5: Running Ablation Experiments"
echo "============================================"
echo "Data directory: ${DATA_DIR}"
echo "Precision: ${PRECISION}"

# Export environment variables for the ablations script
export DATA_DIR="${DATA_DIR}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"

# We cd to REPO_ROOT to ensure script paths work correctly
cd "${REPO_ROOT}"

# Run the ablations
./benchmarks/run_ablations_local.sh "${GPU_ID}"

echo ""
echo "============================================"
echo "Ablations finished successfully!"
echo "============================================"
echo "Results synced to: ${S3_OUTPUT}"
