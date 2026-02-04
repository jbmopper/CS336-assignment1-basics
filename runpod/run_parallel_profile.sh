#!/bin/bash
# Launch parallel profiling jobs on available GPUs
# Usage: ./run_parallel_profile.sh

# Array of profiling scenarios (format: "name args...")
# We use the 'assignment' preset as a base and override parameters
CONFIGS=(
    "bandwidth_bound --batch-size 256 --d-model 384 --num-heads 6 --num-layers 4 --d-ff 1024"
    "compute_bound   --batch-size 32 --d-model 1536 --num-heads 24 --num-layers 8 --d-ff 4096"
    "attn_memory     --batch-size 16 --d-model 512 --num-heads 8 --num-layers 6 --seq-len 2048"
    "vocab_bound     --batch-size 64 --d-model 512 --num-heads 8 --num-layers 4 --vocab-size 50257"
)

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

pids=()

for i in "${!CONFIGS[@]}"; do
    CFG_STR="${CONFIGS[$i]}"
    # Split name and args
    NAME=$(echo $CFG_STR | cut -d' ' -f1)
    ARGS=$(echo $CFG_STR | cut -d' ' -f2-)
    
    # Round-robin GPU assignment
    GPU_ID=$((i % NUM_GPUS))
    
    echo "Launching '$NAME' on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run python -m benchmarks.profile_cuda \
        --model assignment \
        --output-dir "profile_results/parallel/$NAME" \
        --precision bf16 \
        --steps 20 \
        $ARGS > "profile_results/parallel/${NAME}.log" 2>&1 &
    
    pids+=($!)
    
    # Small stagger to avoid thundering herd on disk/PCIe
    sleep 2
done

echo "All jobs launched. Waiting for completion..."
for pid in "${pids[@]}"; do
    wait $pid
done
echo "Done! Results in profile_results/parallel/"
