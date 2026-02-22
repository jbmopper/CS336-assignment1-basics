#!/bin/bash
# Train all model variants with optimized hyperparameters and save checkpoints.
#
# Usage:
#   ./runpod/run_optimized_checkpoints.sh [gpu_id]
#
#   # Run in background (survives SSH disconnect):
#   nohup ./runpod/run_optimized_checkpoints.sh > optimized.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ── Configuration ─────────────────────────────────────────────────────────────

S3_DEFAULT_BUCKET="s3://cs336-spot-287998774376-us-west-2/assignment1-basics"
S3_TOKENIZED="${S3_TOKENIZED:-${S3_DEFAULT_BUCKET}/tokenized/}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
S3_OUTPUT="${S3_OUTPUT:-${S3_DEFAULT_BUCKET}/optimized_checkpoints/${RUN_TS}}"

DATA_DIR="${DATA_DIR:-/workspace/tokenized}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/optimized}"

WANDB_PROJECT="optimized_checkpoints"
WANDB_ENTITY="${WANDB_ENTITY:-jbmopper-0}"

# Optimized hyperparameters from sweep
LR_MAX=0.002457597548470604
LR_MIN=0.0002457597548470604
BETA2=0.9511132832116378
WEIGHT_DECAY=0.08610556049547105

NUM_ITERS=5000
WARMUP_ITERS=500
COSINE_ITERS=5000
EVAL_EVERY=50

# ── Data Setup ────────────────────────────────────────────────────────────────

echo "============================================"
echo "Syncing tokenized data from S3"
echo "============================================"
mkdir -p "${DATA_DIR}"
aws s3 sync "${S3_TOKENIZED}" "${DATA_DIR}" --no-progress

if [[ ! -f "${DATA_DIR}/tinystories_train.npy" || ! -f "${DATA_DIR}/tinystories_valid.npy" ]]; then
    echo "ERROR: Tokenized data not found in ${DATA_DIR}"
    exit 1
fi

# Verify EOS token is present (sanity check for the special token fix)
uv run python -c "
import numpy as np
data = np.load('${DATA_DIR}/tinystories_train.npy', mmap_mode='r')
eos_count = int(np.sum(data == 256))
print(f'EOS token count in train data: {eos_count:,}')
if eos_count == 0:
    raise RuntimeError('FATAL: No EOS tokens found — tokenized data is broken!')
print('Special token check passed.')
"

# Symlink for local code paths
if [[ ! -e "${REPO_ROOT}/tokenized" ]]; then
    ln -sf "${DATA_DIR}" "${REPO_ROOT}/tokenized"
fi

# ── Background S3 sync ───────────────────────────────────────────────────────

mkdir -p "${CHECKPOINT_DIR}"

SYNC_LOG="/tmp/s3sync_optimized.log"
nohup bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" 120 \
    > "${SYNC_LOG}" 2>&1 &
SYNC_PID=$!

cleanup() {
    echo ""
    echo "Stopping background sync..."
    kill "${SYNC_PID}" 2>/dev/null || true
    echo "Final sync..."
    bash "${REPO_ROOT}/runpod/sync_outputs.sh" "${S3_OUTPUT}" "${CHECKPOINT_DIR}" 0 || true
}
trap cleanup EXIT

# ── GPU info ──────────────────────────────────────────────────────────────────

PRECISION="bf16"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${GPU_ID}" || true
else
    echo "WARNING: No CUDA GPU detected, falling back to fp32"
    PRECISION="fp32"
fi

echo ""
echo "============================================"
echo "Optimized Checkpoint Training"
echo "============================================"
echo "W&B Project: ${WANDB_PROJECT}"
echo "Precision:   ${PRECISION}"
echo "LR max:      ${LR_MAX}"
echo "Beta2:       ${BETA2}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Iters:       ${NUM_ITERS}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "S3 Output:   ${S3_OUTPUT}"
echo ""

# ── Common arguments ──────────────────────────────────────────────────────────

COMMON_ARGS="--data-dir ${DATA_DIR} \
    --precision ${PRECISION} \
    --num-iters ${NUM_ITERS} \
    --warmup-iters ${WARMUP_ITERS} \
    --cosine-iters ${COSINE_ITERS} \
    --lr ${LR_MAX} \
    --lr-min ${LR_MIN} \
    --beta2 ${BETA2} \
    --weight-decay ${WEIGHT_DECAY} \
    --eval-every ${EVAL_EVERY} \
    --save-best --save-final \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-entity ${WANDB_ENTITY}"

run_exp() {
    local NAME="$1"
    shift
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "  Running: ${NAME}"
    echo "╚══════════════════════════════════════════╝"
    uv run python -m cs336_basics.train \
        --run-name "${NAME}" \
        --checkpoint-dir "${CHECKPOINT_DIR}/${NAME}" \
        ${COMMON_ARGS} \
        "$@"
}

# ── 1. Baseline (assignment default, 17M params) ─────────────────────────────

run_exp "baseline" \
    --d-model 512 --num-heads 16 --num-layers 4 --d-ff 1344 \
    --batch-size 32

# ── 2. Model I (original intuitive A: wide-attention, 49M params) ─────────

run_exp "model_I" \
    --d-model 640 --num-heads 10 --num-layers 10 --d-ff 1024 \
    --batch-size 64

# ── 3. Model J (original intuitive B: deep FFN-heavy, 39M params) ────────

run_exp "model_J" \
    --d-model 384 --num-heads 12 --num-layers 12 --d-ff 1728 \
    --batch-size 48

# ── 4. Model A wide (shallow-wide, 30M params) ───────────────────────────

run_exp "model_A_wide" \
    --d-model 768 --num-heads 12 --num-layers 2 --d-ff 2048 \
    --batch-size 32

# ── 5. Model B deep (narrow-deep, 29M params) ────────────────────────────

run_exp "model_B_deep" \
    --d-model 384 --num-heads 12 --num-layers 12 --d-ff 1024 \
    --batch-size 32

# ── 6-9. Ablations (all on assignment default arch) ──────────────────────

ABLATION_ARCH="--d-model 512 --num-heads 16 --num-layers 4 --d-ff 1344 --batch-size 32"

run_exp "ablation_no_norm" \
    ${ABLATION_ARCH} --norm-mode none

run_exp "ablation_post_norm" \
    ${ABLATION_ARCH} --norm-mode post

run_exp "ablation_nope" \
    ${ABLATION_ARCH} --no-rope

run_exp "ablation_silu" \
    ${ABLATION_ARCH} --ffn-type silu --ffn-hidden-dim 2048

echo ""
echo "============================================"
echo "All 9 runs complete!"
echo "============================================"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "S3: ${S3_OUTPUT}"
