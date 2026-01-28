#!/bin/bash
# Full profiling of Model A and Model B with Instruments.

TRACE_DIR="/Volumes/slab_storage/traces"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_STEPS=30
TIME_LIMIT=120s

# Check trace directory exists
if [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory does not exist: $TRACE_DIR"
    mkdir -p "$TRACE_DIR" || exit 1
fi

echo "============================================================"
echo "Full Model Profiling - $TIMESTAMP"
echo "============================================================"

# Find the real python path
PYTHON_EXE=$(uv run python -c "import sys; import os; print(os.path.realpath(sys.executable))")
echo "Using Python binary: $PYTHON_EXE"

function run_profile() {
    local model=$1
    local name=$2
    local output="${TRACE_DIR}/model_${model}_${TIMESTAMP}_metal.trace"
    
    echo "------------------------------------------------------------"
    echo "Profiling Model $model ($name)..."
    echo "Output: $output"
    echo "------------------------------------------------------------"
    
    # We use xcrun xctrace record to launch the python process
    # We add --template 'Metal System Trace' and 'Time Profiler'
    xcrun xctrace record \
        --output "$output" \
        --template 'Metal System Trace' \
        --template 'Time Profiler' \
        --time-limit "$TIME_LIMIT" \
        --launch -- "$PYTHON_EXE" -m benchmarks.profile_training \
            --model "$model" \
            --num-steps "$NUM_STEPS" \
            --trace-dir "$TRACE_DIR"
            
    echo "Model $model complete."
}

# Run both models
run_profile "A" "Wide Attention"
sleep 5
run_profile "B" "Standard FFN"

echo "============================================================"
echo "PROFILING COMPLETE"
echo "============================================================"
echo "Traces saved to $TRACE_DIR"
