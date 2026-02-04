#!/bin/bash
# Run local ablations sequentially.
# Usage: ./benchmarks/run_ablations_local.sh [gpu_id]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/tokenized}"

# Common settings
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: Tokenized data not found at ${DATA_DIR}"
    echo "Set DATA_DIR or sync tokenized data."
    exit 1
fi

COMMON_ARGS="--data-dir ${DATA_DIR} --precision bf16 --num-iters 5000 --eval-every 500 --save-every 5000"

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
        --checkpoint-dir "${REPO_ROOT}/checkpoints/ablations/${NAME}" \
        $COMMON_ARGS \
        "$@"
}

# --- Define Experiments Below ---

# 1. Baseline (Assignment Config)
run_exp "baseline" \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --batch-size 64

# 2. Wide (More heads/width, fewer layers)
run_exp "wide" \
    --d-model 768 --num-layers 2 --num-heads 12 --d-ff 2048 --batch-size 64

# 3. Deep (More layers, narrower)
run_exp "deep" \
    --d-model 384 --num-layers 8 --num-heads 12 --d-ff 1024 --batch-size 64

# 4. Large Vocab (Test vocab impact)
run_exp "large_vocab" \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --vocab-size 50000 --batch-size 64

echo ""
echo "All ablations complete!"
