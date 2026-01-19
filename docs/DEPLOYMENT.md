# Deployment Guide

Running training on cloud GPUs or in Docker containers.

---

## Option 1: Cloud GPU Instance (RunPod, Lambda, etc.)

### Providers
| Provider | A100 80GB | Notes |
|----------|-----------|-------|
| Vast.ai | ~$1.50/hr | Cheapest, variable availability |
| RunPod | ~$2/hr | Easy to use, good for beginners |
| Lambda Labs | ~$2.50/hr | Simple, often sold out |
| GCP/AWS | ~$4/hr | More setup, most reliable |

For learning/small models, A100 is plenty. H100 is overkill.

### Steps

1. **Create account** at your chosen provider

2. **Launch instance** with PyTorch template (most providers have this)

3. **SSH in**
   ```bash
   ssh user@instance-ip
   ```

4. **Clone your code**
   ```bash
   git clone <your-repo>
   cd assignment1-basics
   ```

5. **Install uv**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```

6. **Install dependencies**
   ```bash
   uv sync
   ```

7. **Upload data** (if not included in repo)
   ```bash
   # From your local machine:
   scp -r /path/to/data user@instance-ip:/path/to/project/data
   ```

8. **Login to wandb** (one time)
   ```bash
   uv run wandb login
   ```

9. **Run training in tmux** (so it survives SSH disconnect)
   ```bash
   tmux new -s train
   uv run python your_training_script.py
   # Ctrl+b, d to detach
   # tmux attach -t train to reconnect later
   ```

### Tips
- Use `tmux` or `screen` for long-running jobs
- Check GPU usage: `nvidia-smi`
- Monitor with wandb dashboard from anywhere

---

## Option 2: Docker on Linux Box

### Why Docker?
- Memory limits (assignment requires ≤100GB for BPE training)
- Reproducible environment
- Isolation from system Python

### Dockerfile

Use the provided `Dockerfile.tokenize` for tokenizer training and corpus tokenization.

### Build

```bash
docker build -f Dockerfile.tokenize -t cs336-tokenize .
```

### Run with memory limit

**BPE training (CPU, memory-limited):**
```bash
docker run --memory=100g \
  -v /path/to/data:/data:ro \
  -v $(pwd)/tokenizers:/app/tokenizers \
  cs336-tokenize \
  uv run python tokenizing_artifacts/train_owt_tokenizer.py \
    /data/owt_train.txt \
    --vocab-size 32000 \
    --output-dir /app/tokenizers/owt \
    --special-token "<|endoftext|>"
```

**Tokenize corpus (CPU, memory-limited):**
```bash
docker run --memory=100g \
  -v /path/to/data:/data:ro \
  -v $(pwd)/tokenizers:/app/tokenizers \
  -v $(pwd)/tokenized:/app/tokenized \
  cs336-tokenize \
  uv run python tokenizing_artifacts/tokenize_stream.py \
    /data/owt_train.txt \
    --tokenizer-dir /app/tokenizers/owt \
    --output /app/tokenized/owt_train.npy \
    --special-token "<|endoftext|>"
```

**Model training (GPU):**
```bash
docker run --gpus all --memory=100g \
  -v /path/to/data:/data:ro \
  -v $(pwd)/checkpoints:/checkpoints \
  cs336-basics \
  uv run python your_training_script.py
```

### Key flags
- `--memory=100g` - limit RAM to 100GB
- `--gpus all` - enable GPU access (requires nvidia-container-toolkit)
- `-v host:container` - mount directories
- `:ro` - read-only mount

### Check memory usage
```bash
docker stats
```

---

## Data Transfer

### Option A: SCP (simple)
```bash
# Upload tokenized data to cloud instance
scp -r tokenized/ user@instance:/path/to/project/
scp -r tokenizers/ user@instance:/path/to/project/
```

### Option B: Re-tokenize on instance
Just run tokenization as first step. Takes time but no upload needed.

### Option C: Cloud storage (S3/GCS)
```bash
# Upload once
aws s3 sync tokenized/ s3://bucket/tokenized/

# Download on instance
aws s3 sync s3://bucket/tokenized/ tokenized/
```

---

## Checklist

Before running on cloud/Docker:

- [ ] Training script works locally on small data
- [ ] Checkpointing saves/loads correctly
- [ ] Wandb login done (or use `--no-wandb` flag)
- [ ] Data accessible at expected paths
- [ ] Using tmux/screen for long jobs

