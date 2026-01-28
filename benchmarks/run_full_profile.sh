#!/bin/bash
# Full profiling of Model A and Model B with Metal System Traces.
#
# Usage:
#   ./benchmarks/run_full_profile.sh
#
# Output goes to /Volumes/slab_storage/traces/

set -e

TRACE_DIR="/Volumes/slab_storage/traces"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=30
TIME_LIMIT=120s

# Must run from repo root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# Check trace directory exists
if [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory does not exist: $TRACE_DIR"
    echo "Make sure the external drive is mounted."
    exit 1
fi

# Check data exists
if [ ! -f "tokenized/tinystories_train_fixed.npy" ]; then
    echo "ERROR: Tokenized data not found. Run from repository root."
    exit 1
fi

# Verify the profiling script works before running under xctrace
echo "Testing profiling script (quick sanity check)..."
uv run python -c "from benchmarks.profile_training import MODEL_A_CONFIG; print('Import OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: Cannot import profiling script. Check for errors above."
    exit 1
fi

echo "============================================================"
echo "Full Model Profiling - $TIMESTAMP"
echo "============================================================"
echo "Trace directory: $TRACE_DIR"
echo "Steps per model: $NUM_STEPS"
echo "Time limit: $TIME_LIMIT"
echo ""
echo "Estimated time: ~5-6 minutes total"
echo "============================================================"
echo ""

# Create a small wrapper script that xctrace will launch
# This avoids all the bash -lc escaping issues
WRAPPER_A="/tmp/profile_model_a_$$.sh"
WRAPPER_B="/tmp/profile_model_b_$$.sh"

UV_PATH=$(which uv)
echo "Using uv at: $UV_PATH"

cat > "$WRAPPER_A" << EOF
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$(pwd)"
export PYTHONPATH="$(pwd)"
exec "$UV_PATH" run python -m benchmarks.profile_training --model A --num-steps $NUM_STEPS --trace-dir "$TRACE_DIR"
EOF
chmod +x "$WRAPPER_A"

cat > "$WRAPPER_B" << EOF
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
cd "$(pwd)"
export PYTHONPATH="$(pwd)"
exec "$UV_PATH" run python -m benchmarks.profile_training --model B --num-steps $NUM_STEPS --trace-dir "$TRACE_DIR"
EOF
chmod +x "$WRAPPER_B"

# Profile Model A
echo "[1/2] Profiling Model A..."
echo "      Output: ${TRACE_DIR}/model_A_${TIMESTAMP}_metal.trace"
echo ""

xcrun xctrace record \
    --output "${TRACE_DIR}/model_A_${TIMESTAMP}_metal.trace" \
    --template 'Metal System Trace' \
    --time-limit "$TIME_LIMIT" \
    --launch -- "$WRAPPER_A"

echo ""
echo "[1/2] Model A complete."
echo ""

# Pause between models
echo "Waiting 5 seconds..."
sleep 5

# Profile Model B
echo "[2/2] Profiling Model B..."
echo "      Output: ${TRACE_DIR}/model_B_${TIMESTAMP}_metal.trace"
echo ""

xcrun xctrace record \
    --output "${TRACE_DIR}/model_B_${TIMESTAMP}_metal.trace" \
    --template 'Metal System Trace' \
    --time-limit "$TIME_LIMIT" \
    --launch -- "$WRAPPER_B"

echo ""
echo "[2/2] Model B complete."
echo ""

# Cleanup temp scripts
rm -f "$WRAPPER_A" "$WRAPPER_B"

# Summary
echo "============================================================"
echo "PROFILING COMPLETE - $(date)"
echo "============================================================"
echo ""
echo "Metal traces (open with Instruments.app):"
echo "  ${TRACE_DIR}/model_A_${TIMESTAMP}_metal.trace"
echo "  ${TRACE_DIR}/model_B_${TIMESTAMP}_metal.trace"
echo ""
echo "PyTorch traces should also be in: $TRACE_DIR"
ls -lh "${TRACE_DIR}"/*.json 2>/dev/null || echo "  (no .json files found yet)"
echo ""
echo "Disk usage:"
du -sh "${TRACE_DIR}"/model_*_${TIMESTAMP}_metal.trace 2>/dev/null || true
echo ""
echo "To view traces:"
echo "  open '${TRACE_DIR}/model_A_${TIMESTAMP}_metal.trace'"
