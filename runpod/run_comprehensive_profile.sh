#!/bin/bash
# Run comprehensive profiling suite.
# Supports splitting phases across different instances.
#
# Usage: ./run_comprehensive_profile.sh [mode]
# Modes:
#   parallel (or 1) : Run Phase 1 (Parallel NSYS tracing) - Good for multi-GPU nodes
#   serial   (or 2) : Run Phase 2 (Serial NCU analysis)   - Good for single-GPU nodes
#   all             : Run both phases (default)

MODE="${1:-all}"
# set -e removed to allow robust handling of individual failures (OOMs)
# set -e

# Array of profiling scenarios (format: "name args...")
CONFIGS=(
    "assignment      --model assignment"
    "bandwidth_bound --batch-size 256 --d-model 384 --num-heads 6 --num-layers 4 --d-ff 1024"
    "compute_bound   --batch-size 32 --d-model 1536 --num-heads 24 --num-layers 8 --d-ff 4096"
    "attn_memory     --batch-size 16 --d-model 512 --num-heads 8 --num-layers 6 --seq-len 2048"
    "vocab_bound     --batch-size 64 --d-model 512 --num-heads 8 --num-layers 4 --vocab-size 50257"
)

OUTPUT_BASE="profile_results/comprehensive"
mkdir -p "$OUTPUT_BASE"

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

run_phase_1() {
    echo "============================================"
    echo "PHASE 1: Parallel NSYS Tracing"
    echo "============================================"

    pids=()

    for i in "${!CONFIGS[@]}"; do
        CFG_STR="${CONFIGS[$i]}"
        NAME=$(echo $CFG_STR | cut -d' ' -f1)
        ARGS=$(echo $CFG_STR | cut -d' ' -f2-)
        
        # Round robin across available GPUs
        GPU_ID=$((i % NUM_GPUS))
        OUT_DIR="$OUTPUT_BASE/$NAME/nsys"
        mkdir -p "$OUT_DIR"
        
        echo "Launching NSYS '$NAME' on GPU $GPU_ID..."
        
    # Run nsys profile
    # Use || true to prevent script exit on failure (e.g. OOM)
    CUDA_VISIBLE_DEVICES=$GPU_ID nsys profile \
        -t cuda,nvtx \
        -o "$OUT_DIR/trace" \
        --stats=true \
        --force-overwrite=true \
        uv run python -m benchmarks.profile_cuda \
            --model assignment \
            --output-dir "$OUT_DIR" \
            --precision bf16 \
            --steps 20 \
            $ARGS > "$OUT_DIR/run.log" 2>&1 || echo "WARNING: Job $NAME failed (see $OUT_DIR/run.log)" &
            
    pids+=($!)
    sleep 2
done

echo "Waiting for NSYS jobs to complete..."
for pid in "${pids[@]}"; do
    wait $pid || true
done
    echo "Phase 1 Complete."
}

run_phase_2() {
    echo "============================================"
    echo "PHASE 2: Serial NCU Kernel Analysis"
    echo "============================================"
    echo "Note: Running strictly serially on GPU 0 to avoid contention."

    for i in "${!CONFIGS[@]}"; do
        CFG_STR="${CONFIGS[$i]}"
        NAME=$(echo $CFG_STR | cut -d' ' -f1)
        ARGS=$(echo $CFG_STR | cut -d' ' -f2-)
        
        OUT_DIR="$OUTPUT_BASE/$NAME/ncu"
        mkdir -p "$OUT_DIR"
        
        echo "Running NCU '$NAME'..."
        
    # Run ncu profile (1 step only!)
    CUDA_VISIBLE_DEVICES=0 ncu \
        --set full \
        --target-processes all \
        -o "$OUT_DIR/analysis" \
        --force-overwrite \
        uv run python -m benchmarks.profile_cuda \
            --model assignment \
            --output-dir "$OUT_DIR" \
            --precision bf16 \
            --steps 1 \
            --warmup 2 \
            $ARGS > "$OUT_DIR/run.log" 2>&1 || echo "WARNING: NCU Job $NAME failed"
            
    echo "  -> Saved to $OUT_DIR/analysis.ncu-rep"
done
    echo "Phase 2 Complete."
}

if [[ "$MODE" == "parallel" ]] || [[ "$MODE" == "1" ]]; then
    run_phase_1
elif [[ "$MODE" == "serial" ]] || [[ "$MODE" == "2" ]]; then
    run_phase_2
elif [[ "$MODE" == "all" ]]; then
    run_phase_1
    echo ""
    run_phase_2
else
    echo "Unknown mode: $MODE"
    echo "Usage: ./run_comprehensive_profile.sh [parallel|serial|all]"
    exit 1
fi

echo ""
echo "============================================"
echo "Selected profiling complete!"
echo "Results in $OUTPUT_BASE"

