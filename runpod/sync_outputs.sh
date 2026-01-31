#!/usr/bin/env bash
# Periodically sync outputs to S3.
# Usage: ./sync_outputs.sh s3://bucket/path /local/dir [interval_seconds]

set -euo pipefail

S3_DEST="${1:-}"
SRC_DIR="${2:-}"
INTERVAL="${3:-300}"

if [[ -z "${S3_DEST}" || -z "${SRC_DIR}" ]]; then
  echo "Usage: $0 s3://bucket/path /local/dir [interval_seconds]" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install awscli and configure credentials." >&2
  exit 1
fi

EXCLUDES=(
  "--exclude" "*.tmp"
  "--exclude" "*.partial"
  "--exclude" "*.incomplete"
)

sync_once() {
  echo "[sync] $(date -u +"%Y-%m-%dT%H:%M:%SZ") syncing ${SRC_DIR} -> ${S3_DEST}"
  aws s3 sync "${SRC_DIR}" "${S3_DEST}" --no-progress "${EXCLUDES[@]}"
  echo "[sync] done"
}

if [[ "${INTERVAL}" == "0" || "${RUN_ONCE:-0}" == "1" ]]; then
  sync_once
  exit 0
fi

while true; do
  sync_once
  sleep "${INTERVAL}"
done
