#!/bin/bash
# Sync tokenized data from S3 to local workspace
# Usage: ./sync_data.sh [destination_dir]

set -euo pipefail

S3_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics/tokenized/"
DEST_DIR="${1:-/workspace/tokenized}"

echo "Syncing tokenized data from S3..."
echo "  Source: $S3_BUCKET"
echo "  Destination: $DEST_DIR"

mkdir -p "$DEST_DIR"

aws s3 sync "$S3_BUCKET" "$DEST_DIR" --no-progress

echo ""
echo "Data synced successfully!"
ls -lh "$DEST_DIR"

echo ""
echo "To use in training:"
echo "  uv run python -m cs336_basics.train --data-dir $DEST_DIR"
