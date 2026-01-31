#!/bin/bash
# Run architecture sweep benchmark on RunPod
# Usage: ./run_sweep.sh [--smol] [extra_args...]
#
# Examples:
#   ./run_sweep.sh                     # Full sweep
#   ./run_sweep.sh --smol              # Quick test sweep
#   ./run_sweep.sh --max-memory-gb 20  # Custom memory limit

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"

# Check if data exists
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]]; then
    echo "Data not found at $DATA_DIR"
    echo "Run ./runpod/sync_data.sh first"
    exit 1
fi

echo "============================================"
echo "CS336 Architecture Sweep"
echo "============================================"

# Print GPU info
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Copy tokenized data path for benchmark script
export DATA_DIR="$DATA_DIR"

# Change to repo root so benchmark can find tokenized data
cd "$(dirname "$0")/.."

# Create symlink if needed
if [[ ! -d "tokenized" ]] && [[ -d "$DATA_DIR" ]]; then
    ln -sf "$DATA_DIR" tokenized
fi

echo "Starting sweep..."
echo ""

uv run python -m benchmarks.train_benchmark \
    --device cuda \
    --precision bf16 \
    --max-memory-gb 22 \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "============================================"
echo "Sweep complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON results found"
