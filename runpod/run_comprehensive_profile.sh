#!/bin/bash
# Run comprehensive profiling suite (single-GPU oriented).
#
# Usage: ./run_comprehensive_profile.sh [mode]
# Modes:
#   parallel (or 1) : Run Phase 1 (NSYS tracing, serial on GPU 0)
#   serial   (or 2) : Run Phase 2 (NCU analysis, serial on GPU 0)
#   all             : Run both phases (default)

MODE="${1:-all}"
# set -e removed to allow robust handling of individual failures (OOMs)
# set -e

# Single-GPU assumptions
DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
OUTPUT_BASE="profile_results/comprehensive"
mkdir -p "$OUTPUT_BASE"

# Array of profiling scenarios (format: "name args...")
# Configurations validated to fit in 24GB VRAM with memory analysis
# See: notebooks/gpu_model_analysis.py for memory calculations
CONFIGS=(
    # Baseline Models (from configs/models.yaml)
    "model_a           --batch-size 32 --seq-len 256 --d-model 768 --num-heads 12 --num-layers 2 --d-ff 2048"
    "model_b           --batch-size 32 --seq-len 256 --d-model 384 --num-heads 12 --num-layers 12 --d-ff 1024"
    
    # Bottleneck & Stress Tests (sorted by memory usage)
    "latency_bound     --batch-size 1 --seq-len 128 --d-model 512 --num-heads 8 --num-layers 12 --d-ff 1536"
    "misaligned_dims   --batch-size 33 --seq-len 256 --d-model 513 --num-heads 9 --num-layers 6 --d-ff 1537"
    "bad_head_size     --batch-size 32 --seq-len 256 --d-model 672 --num-heads 12 --num-layers 6 --d-ff 1792"
    "vocab_bottleneck  --batch-size 64 --seq-len 256 --d-model 512 --num-heads 8 --num-layers 4 --d-ff 1536 --vocab-size 50257"
    "wide_ffn          --batch-size 32 --seq-len 256 --d-model 768 --num-heads 12 --num-layers 6 --d-ff 4096"
    "bandwidth_bound   --batch-size 256 --seq-len 256 --d-model 384 --num-heads 6 --num-layers 4 --d-ff 1024"
    "compute_bound     --batch-size 32 --seq-len 256 --d-model 1536 --num-heads 24 --num-layers 8 --d-ff 4096"
    "deep_sequential   --batch-size 32 --seq-len 256 --d-model 512 --num-heads 8 --num-layers 32 --d-ff 1536"
)

NSYS_AVAILABLE=1
NCU_AVAILABLE=1
command -v nsys >/dev/null 2>&1 || NSYS_AVAILABLE=0
command -v ncu >/dev/null 2>&1 || NCU_AVAILABLE=0

if [[ $NSYS_AVAILABLE -eq 0 ]]; then
    echo "WARNING: nsys not found in PATH. Phase 1 will fall back to simple timing."
fi
if [[ $NCU_AVAILABLE -eq 0 ]]; then
    echo "WARNING: ncu not found in PATH. Phase 2 will be skipped."
fi

if [[ ! -f "${DATA_DIR}/tinystories_train.npy" ]] && [[ ! -f "${DATA_DIR}/tinystories_train_fixed.npy" ]]; then
    echo "ERROR: Tokenized data not found at ${DATA_DIR}"
    echo "Expected: tinystories_train.npy or tinystories_train_fixed.npy"
    exit 1
fi

WANDB_ARGS=()
WANDB_RUN_NAME_BASE=""
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    if [[ -z "${WANDB_PROJECT:-}" || -z "${WANDB_RUN_NAME:-}" ]]; then
        echo "WARNING: WANDB_ENTITY set but WANDB_PROJECT/WANDB_RUN_NAME missing; W&B disabled."
    else
        WANDB_ARGS=(--wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}")
        WANDB_RUN_NAME_BASE="${WANDB_RUN_NAME}"
    fi
fi

SUCCESS_JOBS=()
FAILED_JOBS=()
SKIPPED_JOBS=()

run_cmd() {
    local name="$1"
    local out_dir="$2"
    shift 2
    local log_file="${out_dir}/run.log"
    local cmd_file="${out_dir}/command.txt"
    local status_file="${out_dir}/status.txt"
    local start_ts
    local end_ts

    start_ts=$(date +%s)
    printf 'command: %s\n' "$*" > "${cmd_file}"

    "$@" > "${log_file}" 2>&1
    local rc=$?
    end_ts=$(date +%s)

    if [[ $rc -eq 0 ]]; then
        printf 'status=ok\nexit_code=%s\nstart_ts=%s\nend_ts=%s\n' "$rc" "$start_ts" "$end_ts" > "${status_file}"
        SUCCESS_JOBS+=("${name}")
    else
        printf 'status=failed\nexit_code=%s\nstart_ts=%s\nend_ts=%s\n' "$rc" "$start_ts" "$end_ts" > "${status_file}"
        FAILED_JOBS+=("${name}")
        echo "WARNING: Job ${name} failed (see ${log_file})"
    fi
    return 0
}

run_phase_1() {
    echo "============================================"
    echo "PHASE 1: NSYS Tracing (serial on GPU 0)"
    echo "============================================"

    for i in "${!CONFIGS[@]}"; do
        CFG_STR="${CONFIGS[$i]}"
        NAME=$(echo $CFG_STR | cut -d' ' -f1)
        ARGS=$(echo $CFG_STR | cut -d' ' -f2-)
        
        OUT_DIR="$OUTPUT_BASE/$NAME/nsys"
        mkdir -p "$OUT_DIR"
        
        echo "Running NSYS '$NAME' on GPU 0..."

        WARGS=()
        if [[ -n "${WANDB_RUN_NAME_BASE}" ]]; then
            WARGS=("${WANDB_ARGS[@]}" --wandb-run-name "${WANDB_RUN_NAME_BASE}-${NAME}-nsys")
        fi
        
        if [[ $NSYS_AVAILABLE -eq 1 ]]; then
            # Run nsys profile (gracefully continue on failure/OOM)
            CUDA_VISIBLE_DEVICES=0 run_cmd "$NAME" "$OUT_DIR" nsys profile \
                -t cuda,nvtx \
                -o "$OUT_DIR/trace" \
                --stats=true \
                --force-overwrite=true \
                uv run python -m benchmarks.profile_cuda \
                    --model assignment \
                    --output-dir "$OUT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --precision bf16 \
                    --steps 20 \
                    --warmup 10 \
                    "${WARGS[@]}" \
                    $ARGS
        else
            echo "NSYS missing - running simple timing for '$NAME'..."
            CUDA_VISIBLE_DEVICES=0 run_cmd "$NAME" "$OUT_DIR" uv run python -m benchmarks.profile_cuda \
                --model assignment \
                --output-dir "$OUT_DIR" \
                --data-dir "$DATA_DIR" \
                --precision bf16 \
                --steps 20 \
                --warmup 10 \
                "${WARGS[@]}" \
                $ARGS
        fi
        sleep 2
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

        WARGS=()
        if [[ -n "${WANDB_RUN_NAME_BASE}" ]]; then
            WARGS=("${WANDB_ARGS[@]}" --wandb-run-name "${WANDB_RUN_NAME_BASE}-${NAME}-ncu")
        fi
        
        if [[ $NCU_AVAILABLE -eq 1 ]]; then
            # Run ncu profile (1 step only with 10 warmup iterations!)
            CUDA_VISIBLE_DEVICES=0 run_cmd "$NAME" "$OUT_DIR" ncu \
                --set detailed \
                --target-processes all \
                -o "$OUT_DIR/analysis" \
                --force-overwrite \
                uv run python -m benchmarks.profile_cuda \
                    --model assignment \
                    --output-dir "$OUT_DIR" \
                    --data-dir "$DATA_DIR" \
                    --precision bf16 \
                    --steps 1 \
                    --warmup 10 \
                    "${WARGS[@]}" \
                    $ARGS
        else
            SKIPPED_JOBS+=("${NAME}")
            echo "WARNING: ncu not available; skipping '$NAME'"
            continue
        fi
            
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
if [[ ${#SUCCESS_JOBS[@]} -eq 0 ]]; then
    echo "Successful jobs: none"
else
    echo "Successful jobs: ${SUCCESS_JOBS[*]}"
fi
if [[ ${#FAILED_JOBS[@]} -eq 0 ]]; then
    echo "Failed jobs: none"
else
    echo "Failed jobs: ${FAILED_JOBS[*]}"
fi
if [[ ${#SKIPPED_JOBS[@]} -eq 0 ]]; then
    echo "Skipped jobs: none"
else
    echo "Skipped jobs: ${SKIPPED_JOBS[*]}"
fi

