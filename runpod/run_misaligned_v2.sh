#!/bin/bash
# One-shot nsys + ncu profiling of a "misaligned but not totally broken" config.
#
# Misalignment budget:
#   batch=34  (not power of 2)
#   seq=257   (prime)
#   d_model=480, num_heads=8, d_head=60  (even but not a tensor-core tile size)
#   d_ff=1538 (odd, not power of 2)
#   num_layers=6
#
# What changed from the original misaligned_dims:
#   d_head: 257 -> 60  (was prime and completely unusable by tensor cores)
#
# Usage:
#   ./runpod/run_misaligned_v2.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
OUTPUT_BASE="profile_results/comprehensive/misaligned_v2"
NAME="misaligned_v2"

# --- S3 sync config ---
S3_DEFAULT_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics"
S3_TOKENIZED="${S3_TOKENIZED:-${S3_DEFAULT_BUCKET}/tokenized}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
S3_OUTPUT="${S3_OUTPUT:-${S3_DEFAULT_BUCKET}/profiling_results/${RUN_TS}}"

COMMON_ARGS="--batch-size 34 --seq-len 257 --d-model 480 --num-heads 8 --num-layers 6 --d-ff 1538"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

has_tokenized_data() {
    local d="$1"
    [[ -f "${d}/tinystories_train.npy" && -f "${d}/tinystories_valid.npy" ]]
}

has_fixed_tokenized_data() {
    local d="$1"
    [[ -f "${d}/tinystories_train_fixed.npy" && -f "${d}/tinystories_valid_fixed.npy" ]]
}

download_tokenized_files() {
    local dest="$1"
    local src="${S3_TOKENIZED%/}"
    mkdir -p "${dest}"
    aws s3 cp "${src}/tinystories_train.npy" "${dest}/tinystories_train.npy" --no-progress
    aws s3 cp "${src}/tinystories_valid.npy" "${dest}/tinystories_valid.npy" --no-progress
}

if ! has_tokenized_data "${DATA_DIR}"; then
    # Try common alternatives on RunPod/local checkouts.
    for candidate in "${REPO_ROOT}/tokenized" "tokenized" "/workspace/tokenized" "/workspace/CS336-assignment1-basics/tokenized"; do
        if has_tokenized_data "${candidate}"; then
            DATA_DIR="${candidate}"
            break
        fi
    done
fi

if ! has_tokenized_data "${DATA_DIR}"; then
    echo "Tokenized data not found locally; attempting S3 pull (TinyStories files only)..."
    echo "  Source: ${S3_TOKENIZED%/}"
    echo "  Destination: ${DATA_DIR}"
    if ! command -v aws &>/dev/null; then
        echo "ERROR: aws CLI not found; cannot download files from S3."
    elif ! aws sts get-caller-identity &>/dev/null; then
        echo "ERROR: AWS credentials invalid or expired; cannot download files from S3."
    elif ! download_tokenized_files "${DATA_DIR}"; then
        echo "ERROR: Failed to download files into ${DATA_DIR}."
    fi
fi

if ! has_tokenized_data "${DATA_DIR}"; then
    echo "ERROR: Tokenized data not found."
    echo "Expected files:"
    echo "  tinystories_train.npy"
    echo "  tinystories_valid.npy"
    echo "Checked DATA_DIR=${DATA_DIR} plus common paths:"
    echo "  ${REPO_ROOT}/tokenized"
    echo "  /workspace/tokenized"
    echo "  /workspace/CS336-assignment1-basics/tokenized"
    if has_fixed_tokenized_data "${DATA_DIR}"; then
        echo "Found only fixed files in DATA_DIR; this script is configured for normal filenames."
    fi
    echo "Set DATA_DIR explicitly, e.g.:"
    echo "  export DATA_DIR=/workspace/CS336-assignment1-basics/tokenized"
    exit 1
fi

export DATA_DIR

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]] && [[ -n "${WANDB_PROJECT:-}" ]] && [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS=(--wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}-${NAME}")
fi

echo "============================================"
echo "Misaligned V2 Profiling"
echo "============================================"
echo "Config: ${COMMON_ARGS}"
echo "  -> d_head = 60 (even, not tensor-core aligned)"
echo "  -> batch=34, seq=257, d_ff=1538 (all non-power-of-2)"
echo "S3 dest: ${S3_OUTPUT}"
echo ""

# --- Verify S3 access early (non-fatal) ---
S3_OK=1
if ! command -v aws &>/dev/null; then
    echo "WARNING: aws CLI not found. Results will NOT be synced to S3."
    S3_OK=0
elif ! aws sts get-caller-identity &>/dev/null; then
    echo "WARNING: AWS credentials invalid or expired. Results will NOT be synced to S3."
    echo "         Run output will still be saved locally at ${OUTPUT_BASE}/"
    S3_OK=0
else
    echo "AWS credentials OK ($(aws sts get-caller-identity --query Arn --output text))"
fi
echo ""

# --- Phase 1: NSYS ---
NSYS_DIR="${OUTPUT_BASE}/nsys"
mkdir -p "${NSYS_DIR}"

if command -v nsys &>/dev/null; then
    echo "[1/2] Running nsys profile (20 steps, 10 warmup)..."
    CUDA_VISIBLE_DEVICES=0 nsys profile \
        -t cuda,nvtx \
        -o "${NSYS_DIR}/trace" \
        --stats=true \
        --force-overwrite=true \
        uv run python -m benchmarks.profile_cuda \
            --model assignment \
            --output-dir "${NSYS_DIR}" \
            --data-dir "${DATA_DIR}" \
            --precision bf16 \
            --steps 20 \
            --warmup 10 \
            "${WANDB_ARGS[@]}" \
            ${COMMON_ARGS}
    echo "[1/2] nsys done -> ${NSYS_DIR}/trace.nsys-rep"
else
    echo "[1/2] nsys not found, running simple timing..."
    CUDA_VISIBLE_DEVICES=0 uv run python -m benchmarks.profile_cuda \
        --model assignment \
        --output-dir "${NSYS_DIR}" \
        --data-dir "${DATA_DIR}" \
        --precision bf16 \
        --steps 20 \
        --warmup 10 \
        "${WANDB_ARGS[@]}" \
        ${COMMON_ARGS}
fi

echo ""
echo "Skipping ncu (no performance counter access)."

echo ""
echo "============================================"
echo "Syncing results to S3"
echo "============================================"
if [[ "${S3_OK}" -eq 1 ]]; then
    REPO_ROOT="$(pwd)"
    bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${OUTPUT_BASE}" 0
    echo "Synced to: ${S3_OUTPUT}"
else
    echo "Skipped S3 sync (no credentials). Results are local at:"
    echo "  $(pwd)/${OUTPUT_BASE}/"
fi

echo ""
echo "============================================"
echo "Misaligned V2 profiling complete!"
echo "Results: ${OUTPUT_BASE}/"
echo "============================================"
