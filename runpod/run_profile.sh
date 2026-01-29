#!/bin/bash
# Run profiling workflows on RunPod
# Usage: ./run_profile.sh [mode] [extra_args...]
#
# Modes:
#   timing      - Simple timing (default, for nsys/ncu)
#   torch       - torch.profiler with Chrome trace
#   nsys        - Full nsys system trace
#   ncu         - ncu kernel analysis (slow, 1 step)
#
# Examples:
#   ./run_profile.sh                          # Simple timing
#   ./run_profile.sh torch                    # torch.profiler
#   ./run_profile.sh nsys --model wide        # nsys with wide model
#   ./run_profile.sh ncu --model assignment   # ncu kernel analysis

set -euo pipefail

MODE="${1:-timing}"
shift 2>/dev/null || true

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
OUTPUT_DIR="${OUTPUT_DIR:-profile_results}"

# Check if data exists
if [[ ! -d "$DATA_DIR" ]] || [[ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]]; then
    echo "Data not found at $DATA_DIR"
    echo "Run ./runpod/sync_data.sh first"
    exit 1
fi

echo "============================================"
echo "CS336 CUDA Profiling"
echo "============================================"
echo "Mode: $MODE"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# Print GPU info
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""

mkdir -p "$OUTPUT_DIR"

# Change to repo root
cd "$(dirname "$0")/.."

# Create symlink if needed
if [[ ! -d "tokenized" ]] && [[ -d "$DATA_DIR" ]]; then
    ln -sf "$DATA_DIR" tokenized
fi

case "$MODE" in
    timing)
        echo "Running simple timing profiling..."
        echo "(Use 'nsys' or 'ncu' mode for detailed GPU analysis)"
        echo ""
        uv run python -m benchmarks.profile_cuda \
            --precision bf16 \
            --output-dir "$OUTPUT_DIR" \
            "$@"
        ;;
    
    torch)
        echo "Running torch.profiler..."
        echo ""
        uv run python -m benchmarks.profile_cuda \
            --precision bf16 \
            --output-dir "$OUTPUT_DIR" \
            --torch-profiler \
            "$@"
        ;;
    
    nsys)
        echo "Running Nsight Systems trace..."
        echo ""
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        nsys profile \
            -t cuda,nvtx \
            -o "$OUTPUT_DIR/nsys_trace_$TIMESTAMP" \
            --stats=true \
            uv run python -m benchmarks.profile_cuda \
                --precision bf16 \
                "$@"
        echo ""
        echo "Trace saved to: $OUTPUT_DIR/nsys_trace_$TIMESTAMP.nsys-rep"
        echo "View with: nsys-ui $OUTPUT_DIR/nsys_trace_$TIMESTAMP.nsys-rep"
        ;;
    
    ncu)
        echo "Running Nsight Compute kernel analysis..."
        echo "(This is slow - analyzing every kernel)"
        echo ""
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ncu --set full \
            --target-processes all \
            -o "$OUTPUT_DIR/ncu_analysis_$TIMESTAMP" \
            uv run python -m benchmarks.profile_cuda \
                --precision bf16 \
                --steps 1 \
                "$@"
        echo ""
        echo "Analysis saved to: $OUTPUT_DIR/ncu_analysis_$TIMESTAMP.ncu-rep"
        echo "View with: ncu-ui $OUTPUT_DIR/ncu_analysis_$TIMESTAMP.ncu-rep"
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: timing, torch, nsys, ncu"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Profiling complete!"
echo "============================================"
