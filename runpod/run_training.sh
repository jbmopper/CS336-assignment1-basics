#!/bin/bash
# Run training on RunPod with sensible defaults
# Usage: ./run_training.sh [run_name] [extra_args...]
#
# Examples:
#   ./run_training.sh                     # Default training run
#   ./run_training.sh test-run            # Named run
#   ./run_training.sh ablation --num-iters 1000 --lr 5e-4

set -euo pipefail

RUN_NAME="${1:-runpod-$(date +%Y%m%d_%H%M)}"
shift 2>/dev/null || true  # Remove run_name from args if provided

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"

# Check if data exists
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]]; then
    echo "Data not found at $DATA_DIR"
    echo "Run ./runpod/sync_data.sh first"
    exit 1
fi

echo "============================================"
echo "CS336 Training Run: $RUN_NAME"
echo "============================================"
echo "Data: $DATA_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo ""

# Print GPU info
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available!')
"

echo ""
echo "Starting training..."
echo ""

uv run python -m cs336_basics.train \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --precision bf16 \
    --batch-size 64 \
    --run-name "$RUN_NAME" \
    "$@"

echo ""
echo "============================================"
echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "============================================"
