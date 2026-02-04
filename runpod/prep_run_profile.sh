#!/usr/bin/env bash
# Prepare environment, sync data, and run comprehensive profiling with background S3 sync.
#
# Usage:
#   ./runpod/prep_run_profile.sh [s3_output_url] [mode]
#
# Arguments:
#   s3_output_url : S3 path to save results (default: s3://.../profiling_results)
#   mode          : Profiling mode (parallel, serial, all) - default: all
#
# Example:
#   ./runpod/prep_run_profile.sh s3://my-bucket/profiles parallel

set -euo pipefail

# --- Configuration ---
S3_DEFAULT_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics"
S3_OUTPUT="${1:-${S3_DEFAULT_BUCKET}/profiling_results/$(date +%Y%m%d_%H%M%S)}"
MODE="${2:-all}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_TOKENIZED="${S3_DEFAULT_BUCKET}/tokenized/"
DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
TOKENIZED_SYMLINK="${REPO_ROOT}/tokenized"
RESULTS_DIR="${REPO_ROOT}/profile_results/comprehensive"

# --- 1. Data Setup ---
echo "============================================"
echo "Step 1: Syncing Data"
echo "============================================"
echo "Downloading tokenized data from: ${S3_TOKENIZED}"
echo "Destination: ${DATA_DIR}"

mkdir -p "${DATA_DIR}"
aws s3 sync "${S3_TOKENIZED}" "${DATA_DIR}" --no-progress

# Create symlink for local scripts
if [[ ! -e "${TOKENIZED_SYMLINK}" ]]; then
    ln -sf "${DATA_DIR}" "${TOKENIZED_SYMLINK}"
    echo "Created symlink: ${TOKENIZED_SYMLINK} -> ${DATA_DIR}"
fi

# Ensure expected filenames exist (legacy support)
if [[ -f "${TOKENIZED_SYMLINK}/tinystories_train.npy" ]]; then
    ln -sf "tinystories_train.npy" "${TOKENIZED_SYMLINK}/tinystories_train_fixed.npy" 2>/dev/null || true
fi
if [[ -f "${TOKENIZED_SYMLINK}/tinystories_valid.npy" ]]; then
    ln -sf "tinystories_valid.npy" "${TOKENIZED_SYMLINK}/tinystories_valid_fixed.npy" 2>/dev/null || true
fi

# --- 2. Output Sync Setup ---
echo ""
echo "============================================"
echo "Step 2: Starting Background Sync"
echo "============================================"
echo "Local Results: ${RESULTS_DIR}"
echo "Remote S3:     ${S3_OUTPUT}"

mkdir -p "${RESULTS_DIR}"

# Start sync script in background (sync every 30s)
nohup bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${RESULTS_DIR}" 30 \
    > /tmp/s3sync_profile.log 2>&1 &
SYNC_PID=$!
echo "Background sync started (PID: ${SYNC_PID})"

# Register cleanup trap to ensure final sync
cleanup() {
    echo ""
    echo "============================================"
    echo "Stopping background sync..."
    kill "${SYNC_PID}" || true
    
    echo "Performing FINAL sync..."
    bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${RESULTS_DIR}" 0
    echo "Done."
}
trap cleanup EXIT

# --- 3. Run Profiling ---
echo ""
echo "============================================"
echo "Step 3: Running Comprehensive Profiling"
echo "============================================"
echo "Mode: ${MODE}"

# We cd to REPO_ROOT to ensure script paths work correctly
cd "${REPO_ROOT}"

# Run the profiling suite
# Note: output directory is hardcoded in run_comprehensive_profile.sh to match RESULTS_DIR
./runpod/run_comprehensive_profile.sh "${MODE}"

echo ""
echo "Profiling finished successfully."
