#!/bin/bash
# Full profiling of Model A and Model B with Metal System Traces.
# Run this and go take a shower.
#
# Usage:
#   ./benchmarks/run_full_profile.sh
#
# Output goes to /Volumes/slab_storage/traces/

set -e  # Exit on error

TRACE_DIR="/Volumes/slab_storage/traces"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=30
TIME_LIMIT=90s

# Check trace directory exists
if [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory does not exist: $TRACE_DIR"
    echo "Make sure the external drive is mounted."
    exit 1
fi

echo "============================================================"
echo "Full Model Profiling - $(date)"
echo "============================================================"
echo "Trace directory: $TRACE_DIR"
echo "Steps per model: $NUM_STEPS"
echo "Time limit per model: $TIME_LIMIT"
echo ""
echo "Estimated time: ~5-6 minutes total"
echo "Estimated disk: ~2-4 GB total"
echo "============================================================"
echo ""

# Get the actual python path so xctrace can follow it (uv run spawns a child)
PYTHON_EXE=$(uv run python -c "import sys; print(sys.executable)")

# Profile Model A
echo "[1/2] Profiling Model A..."
echo "      Output: ${TRACE_DIR}/model_a_${TIMESTAMP}_metal.trace"
echo ""

xcrun xctrace record \
    --output "${TRACE_DIR}/model_a_${TIMESTAMP}_metal.trace" \
    --template 'Metal System Trace' \
    --template 'Time Profiler' \
    --time-limit "$TIME_LIMIT" \
    --launch -- "$PYTHON_EXE" -m benchmarks.profile_training \
        --model A \
        --num-steps "$NUM_STEPS" \
        --trace-dir "$TRACE_DIR"

echo ""
echo "[1/2] Model A complete."
echo ""

# Small pause to let things settle
sleep 5

# Profile Model B
echo "[2/2] Profiling Model B..."
echo "      Output: ${TRACE_DIR}/model_b_${TIMESTAMP}_metal.trace"
echo ""

xcrun xctrace record \
    --output "${TRACE_DIR}/model_b_${TIMESTAMP}_metal.trace" \
    --template 'Metal System Trace' \
    --template 'Time Profiler' \
    --time-limit "$TIME_LIMIT" \
    --launch -- "$PYTHON_EXE" -m benchmarks.profile_training \
        --model B \
        --num-steps "$NUM_STEPS" \
        --trace-dir "$TRACE_DIR"

echo ""
echo "[2/2] Model B complete."
echo ""

# Summary
echo "============================================================"
echo "PROFILING COMPLETE - $(date)"
echo "============================================================"
echo ""
echo "Metal traces (open with Instruments.app):"
echo "  ${TRACE_DIR}/model_a_${TIMESTAMP}_metal.trace"
echo "  ${TRACE_DIR}/model_b_${TIMESTAMP}_metal.trace"
echo ""
echo "PyTorch traces (load in chrome://tracing):"
ls -la "${TRACE_DIR}"/Model_*_torch_trace.json 2>/dev/null || echo "  (check ${TRACE_DIR} for .json files)"
echo ""
echo "Disk usage:"
du -sh "${TRACE_DIR}"/*.trace 2>/dev/null || true
echo ""
echo "To compare in Instruments:"
echo "  open '${TRACE_DIR}/model_a_${TIMESTAMP}_metal.trace'"
echo "  open '${TRACE_DIR}/model_b_${TIMESTAMP}_metal.trace'"
