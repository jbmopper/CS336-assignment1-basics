#!/usr/bin/env bash
# Sync tokenized data, add symlinks for train_compare, and start S3 output sync.
#
# Usage:
#   ./runpod/prep_train_compare.sh s3://your-bucket/outputs [interval_seconds]
#
# Optional env vars:
#   S3_TOKENIZED   - S3 source for tokenized data
#   DATA_DIR       - local tokenized data directory (default: /workspace/tokenized)
#   CHECKPOINT_DIR - local checkpoint dir (default: /workspace/checkpoints/model_comp)

set -euo pipefail

S3_OUTPUT="${1:-}"
INTERVAL="${2:-300}"

if [[ -z "${S3_OUTPUT}" ]]; then
  echo "Usage: $0 s3://your-bucket/outputs [interval_seconds]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S3_TOKENIZED="${S3_TOKENIZED:-s3://cs336-spot-287998774376-us-west-2/assignment1-basics/tokenized/}"
DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
TOKENIZED_DIR="${REPO_ROOT}/tokenized"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/model_comp}"

echo "Syncing tokenized data from S3..."
echo "  Source: ${S3_TOKENIZED}"
echo "  Dest:   ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
aws s3 sync "${S3_TOKENIZED}" "${DATA_DIR}" --no-progress

if [[ ! -e "${TOKENIZED_DIR}" ]]; then
  ln -s "${DATA_DIR}" "${TOKENIZED_DIR}"
fi

# Ensure expected filenames for train_compare.py
if [[ -f "${TOKENIZED_DIR}/tinystories_train.npy" ]]; then
  ln -sf "tinystories_train.npy" "${TOKENIZED_DIR}/tinystories_train_fixed.npy"
fi
if [[ -f "${TOKENIZED_DIR}/tinystories_valid.npy" ]]; then
  ln -sf "tinystories_valid.npy" "${TOKENIZED_DIR}/tinystories_valid_fixed.npy"
fi

mkdir -p "${CHECKPOINT_DIR}"

echo "Starting background S3 sync for checkpoints..."
echo "  Local: ${CHECKPOINT_DIR}"
echo "  Remote: ${S3_OUTPUT}"
nohup bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" "${INTERVAL}" \
  > /tmp/s3sync_model_comp.log 2>&1 &
echo "  Log: /tmp/s3sync_model_comp.log"

echo ""
echo "Ready to run:"
echo "  cd \"${REPO_ROOT}\""
echo "  uv run python -m cs336_basics.train_compare --precision bf16 --checkpoint-dir \"${CHECKPOINT_DIR}\""
