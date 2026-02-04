# RunPod GPU Training Setup

Deploy training and profiling workloads on RunPod with a 4090 GPU.

## Prerequisites

1. **RunPod account**: https://runpod.io
2. **AWS CLI** (for S3 data access): `brew install awscli` or `pip install awscli`
3. **runpodctl** (optional, for file transfers): https://github.com/runpod/runpodctl

## Quick Start

### 1. Create a Pod

Go to RunPod Console → Deploy → Select:
- **GPU**: RTX 4090 (24GB VRAM)
- **Template**: PyTorch 2.x (or RunPod PyTorch)
- **Container Disk**: 50GB (for code + checkpoints)
- **Volume Disk**: 0GB (we'll use S3 for data)

Recommended specs:
- Community Cloud: ~$0.50-0.70/hr
- Secure Cloud: ~$0.80/hr
- RAM: 32GB+ recommended

### 2. Connect to Pod

```bash
# Via web terminal (easiest)
# Click "Connect" → "Start Web Terminal"

# Or via SSH (if you added SSH key)
ssh root@<pod-ip> -p <ssh-port>
```

### 3. Setup Environment

Run once after pod creation:

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/CS336-assignment1-basics.git
cd CS336-assignment1-basics

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install dependencies
uv sync

# Configure AWS (for S3 data access)
aws configure
# Enter your AWS credentials when prompted
```

### 4. Sync Data from S3

```bash
# Download tokenized data
./runpod/sync_data.sh

# Or manually:
aws s3 sync s3://cs336-spot-287998774376-us-west-2/assignment1-basics/tokenized/ /workspace/tokenized/
```

### 5. Run Training

```bash
# Quick test (10 iterations, no W&B)
uv run python -m cs336_basics.train \
    --data-dir /workspace/tokenized \
    --precision bf16 \
    --batch-size 64 \
    --num-iters 10 \
    --no-wandb

# Full training run
uv run python -m cs336_basics.train \
    --data-dir /workspace/tokenized \
    --precision bf16 \
    --batch-size 64 \
    --num-iters 5000 \
    --run-name "4090-bf16-run1"
```

### 6. Run Benchmarks

```bash
# Architecture sweep (grid search)
uv run python -m benchmarks.train_benchmark \
    --device cuda \
    --precision bf16 \
    --max-memory-gb 22 \
    --output-dir benchmark_results

# Quick sweep for testing
uv run python -m benchmarks.train_benchmark \
    --device cuda \
    --precision bf16 \
    --smol
```

### 7. Run Profiling

```bash
# Simple timing (compatible with nsys/ncu)
uv run python -m benchmarks.profile_cuda \
    --model assignment \
    --precision bf16 \
    --steps 10

# With torch.profiler (Chrome trace output)
uv run python -m benchmarks.profile_cuda \
    --model assignment \
    --precision bf16 \
    --steps 10 \
    --torch-profiler

# Full nsys trace
nsys profile -t cuda,nvtx -o training_trace --stats=true \
    uv run python -m benchmarks.profile_cuda --model assignment --precision bf16

# Kernel-level analysis with ncu
ncu --set full --target-processes all -o kernel_analysis \
    uv run python -m benchmarks.profile_cuda --model assignment --precision bf16 --steps 1
```

### 8. Run Ablation Studies

Run architecture ablations (wide vs deep models, FFN scaling, etc.) with automatic S3 sync:

```bash
# All-in-one: data sync + ablations + checkpoint backup
./runpod/prep_run_ablations.sh s3://YOUR-BUCKET/ablations 0

# Or with default S3 bucket
./runpod/prep_run_ablations.sh

# Custom W&B project
WANDB_PROJECT="cs336-ablations" ./runpod/prep_run_ablations.sh

# Manual run (after data setup)
export DATA_DIR=/workspace/tokenized
export CHECKPOINT_DIR=/workspace/checkpoints/ablations
export PRECISION=bf16
./benchmarks/run_ablations_local.sh 0
```

The prep script will:
- Download tokenized data from S3
- Create necessary symlinks
- Set up checkpoints directory
- Start background S3 sync (every 60s, skips `latest.pt` to reduce overhead)
- Run all ablation experiments sequentially with W&B logging
- Save best and final checkpoints (synced to S3)
- Perform final sync on completion

**W&B Logging:** Each ablation is logged to Weights & Biases with run names:
- `ablation_baseline` - Pre-norm + RoPE + SwiGLU (default assignment config)
- `ablation_no_norm` - No layer normalization (norm_mode=none)
- `ablation_post_norm` - Post-norm instead of pre-norm
- `ablation_nope` - No position embeddings (NoPE)
- `ablation_silu` - SiLU FFN (non-gated) instead of SwiGLU (gated)

## Data Location

Tokenized data is stored in S3:
```
s3://cs336-spot-287998774376-us-west-2/assignment1-basics/tokenized/
├── tinystories_train.npy
├── tinystories_valid.npy
├── owt_train.npy (if available)
└── owt_valid.npy (if available)
```

## File Transfer

### Download results from pod

```bash
# Using runpodctl (if installed)
runpodctl receive <file-or-dir>

# Or sync to S3
aws s3 sync checkpoints/ s3://YOUR-BUCKET/checkpoints/
aws s3 sync benchmark_results/ s3://YOUR-BUCKET/benchmark_results/

# Or use SCP (if SSH enabled)
scp -P <ssh-port> -r root@<pod-ip>:/workspace/CS336-assignment1-basics/checkpoints/ ./
```

## Cost Optimization

1. **Stop pod when not in use** - You're charged per hour while running
2. **Use Community Cloud** - ~30% cheaper than Secure Cloud
3. **Use tmux for long jobs** - So you can disconnect SSH

```bash
# Start tmux session
tmux new -s train

# Run training
uv run python -m cs336_basics.train ...

# Detach: Ctrl+b, then d
# Reattach later: tmux attach -t train
```

## Troubleshooting

### OOM (Out of Memory)

Reduce batch size or use smaller model:
```bash
uv run python -m cs336_basics.train --batch-size 32
```

### CUDA not found

Make sure you're on a GPU pod and CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

### Slow data loading

Data is memory-mapped from S3. For faster I/O, copy to local SSD:
```bash
cp /workspace/tokenized/*.npy /tmp/
uv run python -m cs336_basics.train --data-dir /tmp
```

## Docker Build (Optional)

If you want to use the custom Docker image:

```bash
# Build locally
docker build -f Dockerfile.train -t your-registry/cs336-train:latest .

# Push to registry (Docker Hub, NGC, etc.)
docker push your-registry/cs336-train:latest

# Use in RunPod: set custom Docker image when creating pod
```
