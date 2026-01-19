#!/bin/bash
# Run this on the EC2 instance after uploading code
# Usage: ./run_on_instance.sh

set -euo pipefail

# Source environment variables (SSM uses non-login shells)
source /etc/profile.d/cs336.sh 2>/dev/null || true

cd /data/assignment1-basics

echo "=== Downloading OWT data ==="
mkdir -p data
cd data
if [[ ! -f owt_train.txt ]]; then
    echo "Downloading owt_train.txt.gz..."
    wget -q --show-progress https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    echo "Extracting..."
    gunzip owt_train.txt.gz
else
    echo "owt_train.txt already exists, skipping download"
fi
if [[ ! -f owt_valid.txt ]]; then
    echo "Downloading owt_valid.txt.gz..."
    wget -q --show-progress https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    echo "Extracting..."
    gunzip owt_valid.txt.gz
else
    echo "owt_valid.txt already exists, skipping download"
fi
cd ..

echo ""
echo "=== Building Docker image ==="
sudo docker build -t cs336-tokenize -f Dockerfile.tokenize .

echo ""
echo "=== Starting tokenizer training in Docker (100GB RAM limit) ==="
mkdir -p tokenizers
echo "Started at: $(date)"
echo ""

# Run in Docker with 100GB memory limit
# Mount data and tokenizers directories for input/output
time sudo docker run \
    --memory=100g \
    --memory-swap=100g \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/tokenizers:/app/tokenizers" \
    cs336-tokenize \
    uv run python tokenizing_artifacts/train_owt_tokenizer.py /app/data/owt_train.txt \
        --vocab-size 32000 \
        --output-dir /app/tokenizers/owt

echo ""
echo "=== Tokenizer training complete! ==="
echo "Finished at: $(date)"
echo ""

echo "=== Tokenizing train dataset ==="
mkdir -p tokenized
echo "Started at: $(date)"

time sudo docker run \
    --memory=100g \
    --memory-swap=100g \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/tokenizers:/app/tokenizers:ro" \
    -v "$(pwd)/tokenized:/app/tokenized" \
    cs336-tokenize \
    uv run python tokenizing_artifacts/tokenize_stream.py /app/data/owt_train.txt \
        --tokenizer-dir /app/tokenizers/owt \
        --output /app/tokenized/owt_train.npy

echo ""
echo "=== Tokenizing validation dataset ==="
echo "Started at: $(date)"

time sudo docker run \
    --memory=100g \
    --memory-swap=100g \
    -v "$(pwd)/data:/app/data:ro" \
    -v "$(pwd)/tokenizers:/app/tokenizers:ro" \
    -v "$(pwd)/tokenized:/app/tokenized" \
    cs336-tokenize \
    uv run python tokenizing_artifacts/tokenize_stream.py /app/data/owt_valid.txt \
        --tokenizer-dir /app/tokenizers/owt \
        --output /app/tokenized/owt_valid.npy

echo ""
echo "=== All tasks complete! ==="
echo "Finished at: $(date)"
echo ""
echo "Results saved to:"
echo "  - tokenizers/owt/ (vocab.json, merges.pkl)"
echo "  - tokenized/owt_train.npy"
echo "  - tokenized/owt_valid.npy"
echo ""

# Upload results to S3 if bucket is configured
if [[ -n "${S3_BUCKET:-}" ]]; then
    echo "Uploading results to S3..."
    aws s3 sync tokenizers/ "s3://$S3_BUCKET/assignment1-basics/tokenizers/"
    aws s3 sync tokenized/ "s3://$S3_BUCKET/assignment1-basics/tokenized/"
    echo ""
    echo "Results uploaded to S3. Download locally with:"
    echo "  aws s3 sync s3://$S3_BUCKET/assignment1-basics/tokenizers ./tokenizers-from-aws"
    echo "  aws s3 sync s3://$S3_BUCKET/assignment1-basics/tokenized ./tokenized-from-aws"
else
    echo "To upload results to S3, run:"
    echo "  aws s3 sync tokenizers/ s3://YOUR-BUCKET/assignment1-basics/tokenizers/"
    echo "  aws s3 sync tokenized/ s3://YOUR-BUCKET/assignment1-basics/tokenized/"
fi
echo ""
echo "Don't forget to terminate the instance when done!"
