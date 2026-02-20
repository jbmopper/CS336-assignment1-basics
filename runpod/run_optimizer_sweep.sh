#!/bin/bash
# Run W&B optimizer sweep on RunPod.
#
# Usage:
#   # Create sweep and start agent (first time):
#   ./runpod/run_optimizer_sweep.sh
#
#   # Join existing sweep (pass sweep ID from wandb output):
#   ./runpod/run_optimizer_sweep.sh <sweep_id>
#
#   # Run in background (survives SSH disconnect):
#   nohup ./runpod/run_optimizer_sweep.sh [sweep_id] > sweep.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
WANDB_ENTITY="${WANDB_ENTITY:-jbmopper-0}"
WANDB_PROJECT="cs336-optimizer-sweep"
SWEEP_CONFIG="cs336_basics/optimizer_sweep_config.yaml"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/checkpoints/optimizer_sweeps}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"

S3_DEFAULT_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics"
S3_TOKENIZED="${S3_TOKENIZED:-${S3_DEFAULT_BUCKET}/tokenized}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
S3_OUTPUT="${S3_OUTPUT:-${S3_DEFAULT_BUCKET}/optimizer_sweeps/${RUN_TS}}"

has_normal_tokenized_pair() {
    local d="$1"
    [[ -f "${d}/tinystories_train.npy" && -f "${d}/tinystories_valid.npy" ]]
}

has_fixed_tokenized_pair() {
    local d="$1"
    [[ -f "${d}/tinystories_train_fixed.npy" && -f "${d}/tinystories_valid_fixed.npy" ]]
}

has_tokenized_data() {
    local d="$1"
    has_normal_tokenized_pair "${d}" || has_fixed_tokenized_pair "${d}"
}

download_tokenized_data() {
    local dest="$1"
    local src="${S3_TOKENIZED%/}"
    mkdir -p "${dest}"

    if aws s3 cp "${src}/tinystories_train.npy" "${dest}/tinystories_train.npy" --no-progress \
        && aws s3 cp "${src}/tinystories_valid.npy" "${dest}/tinystories_valid.npy" --no-progress; then
        return 0
    fi

    # Fallback for older bucket layouts.
    aws s3 cp "${src}/tinystories_train_fixed.npy" "${dest}/tinystories_train_fixed.npy" --no-progress \
        && aws s3 cp "${src}/tinystories_valid_fixed.npy" "${dest}/tinystories_valid_fixed.npy" --no-progress
}

# ── Pre-flight checks ────────────────────────────────────────────────────────

if ! has_tokenized_data "${DATA_DIR}"; then
    for candidate in "${REPO_ROOT}/tokenized" "tokenized" "/workspace/tokenized" "/workspace/CS336-assignment1-basics/tokenized"; do
        if has_tokenized_data "${candidate}"; then
            DATA_DIR="${candidate}"
            break
        fi
    done
fi

if ! has_tokenized_data "${DATA_DIR}"; then
    echo "Tokenized data not found locally; attempting S3 pull..."
    echo "  Source: ${S3_TOKENIZED%/}"
    echo "  Destination: ${DATA_DIR}"
    if ! command -v aws &>/dev/null; then
        echo "ERROR: aws CLI not found; cannot download tokenized data."
    elif ! aws sts get-caller-identity &>/dev/null; then
        echo "ERROR: AWS credentials invalid or expired; cannot download tokenized data."
    elif ! download_tokenized_data "${DATA_DIR}"; then
        echo "ERROR: Failed to download tokenized data from ${S3_TOKENIZED%/}."
    fi
fi

if ! has_tokenized_data "${DATA_DIR}"; then
    echo "ERROR: Tokenized data not found."
    echo "Expected either:"
    echo "  tinystories_train.npy + tinystories_valid.npy"
    echo "or:"
    echo "  tinystories_train_fixed.npy + tinystories_valid_fixed.npy"
    echo "Set DATA_DIR explicitly, e.g.:"
    echo "  export DATA_DIR=/workspace/CS336-assignment1-basics/tokenized"
    exit 1
fi
export DATA_DIR

# Symlink for any code that expects ./tokenized/
if [[ ! -d "tokenized" ]] && [[ -d "${DATA_DIR}" ]] && [[ "${DATA_DIR}" != "tokenized" ]]; then
    ln -sf "${DATA_DIR}" tokenized
fi

mkdir -p "${CHECKPOINT_DIR}"

# Verify wandb is authenticated
if ! uv run python -c "import wandb; wandb.Api()" 2>/dev/null; then
    echo "ERROR: wandb not authenticated. Set WANDB_API_KEY or run 'wandb login'."
    exit 1
fi

# GPU info
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No CUDA GPU detected')
" 2>/dev/null || true

echo ""
echo "============================================"
echo "CS336 Optimizer Sweep"
echo "============================================"
echo "Entity:  ${WANDB_ENTITY}"
echo "Project: ${WANDB_PROJECT}"
echo "Data:    ${DATA_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "S3 Output: ${S3_OUTPUT}"
echo ""

# ── Start background S3 sync ─────────────────────────────────────────────────

SYNC_LOG="/tmp/s3sync_optimizer_sweep.log"
nohup bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" "${SYNC_INTERVAL}" \
    > "${SYNC_LOG}" 2>&1 &
SYNC_PID=$!
echo "Background sync started (PID: ${SYNC_PID})"
echo "Sync log: ${SYNC_LOG}"
echo ""

cleanup() {
    echo ""
    echo "Stopping background sync..."
    kill "${SYNC_PID}" 2>/dev/null || true
    echo "Performing FINAL sync..."
    bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" 0 || true
}
trap cleanup EXIT

# ── Create or join sweep ─────────────────────────────────────────────────────

SWEEP_ID="${1:-}"

if [[ -z "${SWEEP_ID}" ]]; then
    echo "Creating new sweep from ${SWEEP_CONFIG}..."
    SWEEP_ID=$(uv run wandb sweep "${SWEEP_CONFIG}" 2>&1 | grep -oP '[\w-]+/[\w-]+/[\w]+$' || true)

    if [[ -z "${SWEEP_ID}" ]]; then
        echo "Could not parse sweep ID. Creating sweep manually..."
        uv run wandb sweep "${SWEEP_CONFIG}"
        echo ""
        echo "Copy the sweep ID from above and re-run:"
        echo "  ./runpod/run_optimizer_sweep.sh <sweep_id>"
        exit 0
    fi
    echo "Created sweep: ${SWEEP_ID}"
else
    echo "Joining existing sweep: ${SWEEP_ID}"
fi

echo ""
echo "Starting sweep agent..."
echo "  (Ctrl+C or kill to stop; the sweep persists on wandb.ai)"
echo ""

uv run wandb agent "${SWEEP_ID}"
