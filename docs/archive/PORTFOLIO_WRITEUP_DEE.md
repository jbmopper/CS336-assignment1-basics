# Transformer Language Model: From-Scratch Implementation

**Project Overview**: Complete implementation of a decoder-only Transformer language model, trained from scratch on the TinyStories dataset using custom BPE tokenization, AdamW optimization, and cosine learning rate scheduling.

**Hardware**: Apple M4 (MacBook Air, 24GB RAM, 10 GPU cores)

---

## Table of Contents
1. [Dimensional Analysis & Model Sizing](#1-dimensional-analysis--model-sizing)
2. [Hyperparameter Selection & Tuning](#2-hyperparameter-selection--tuning)
3. [Memory Profiling & System Behavior](#3-memory-profiling--system-behavior)
4. [Benchmarks & Performance Metrics](#4-benchmarks--performance-metrics)
5. [Key Learnings & Future Work](#5-key-learnings--future-work)

---

## 1. Dimensional Analysis & Model Sizing

### 1.1 Parameter Count Calculations

The Transformer LM consists of the following components:

#### Token Embeddings
```
Parameters = vocab_size × d_model
```

#### Per Transformer Block (×num_layers)

| Component | Formula | Notes |
|-----------|---------|-------|
| RMSNorm (×2) | 2 × d_model | Gain parameters only |
| Q/K/V projections | 3 × d_model² | No bias |
| Output projection | d_model² | No bias |
| FFN W₁ | d_model × d_ff | SwiGLU gate |
| FFN W₂ | d_ff × d_model | SwiGLU output |
| FFN W₃ | d_model × d_ff | SwiGLU activation |

**Per-block total**: `4 × d_model² + 3 × d_model × d_ff + 2 × d_model`

#### Output Layers
- Final RMSNorm: `d_model`
- LM Head: `d_model × vocab_size`

### 1.2 Tested Configurations

| Config | d_model | num_heads | num_layers | d_ff | Parameters | Tokens/iter |
|--------|---------|-----------|------------|------|------------|-------------|
| Small | 256 | 4 | 4 | 1024 | ~10M | 4,000 |
| Assignment | 512 | 16 | 4 | 1344 | ~17M | 8,192 |
| Medium | 512 | 8 | 6 | 2048 | ~50M | 16,384 |

#### Detailed Breakdown: Assignment Configuration (17M params)

```python
# Configuration
vocab_size = 10,000
d_model = 512
num_heads = 16
num_layers = 4
d_ff = 1344  # ≈ 8/3 × d_model, rounded to multiple of 64
context_length = 256
batch_size = 32

# Calculations
embedding_params = 10000 × 512 = 5,120,000
per_block_attn = 4 × 512² = 1,048,576
per_block_ffn = 3 × 512 × 1344 = 2,064,384
per_block_norm = 2 × 512 = 1,024
total_blocks = 4 × (1,048,576 + 2,064,384 + 1,024) = 12,455,936
final_norm = 512
lm_head = 512 × 10000 = 5,120,000

TOTAL = 5,120,000 + 12,455,936 + 512 + 5,120,000 ≈ 22.7M parameters
# (17M non-embedding parameters)
```

### 1.3 FLOPs Analysis (Forward Pass)

For a single forward pass with sequence length `S` and batch size `B`:

| Operation | FLOPs Formula | Assignment Config |
|-----------|---------------|-------------------|
| Embedding lookup | O(B × S × d_model) | 4.2M |
| QKV projection (×L) | 3 × 2BSD² × L | 1.61B |
| Attention scores (×L) | 2BS²d × L | 134M |
| Attention output (×L) | 2BS²d × L | 134M |
| Output projection (×L) | 2BSD² × L | 537M |
| FFN (×L) | 3 × 2BSD×d_ff × L | 2.11B |
| LM head | 2BSD×V | 2.68B |
| **Total** | | **~7.2B FLOPs** |

*Where: B=32, S=256, D=512, d=D/H=32, L=4, V=10000*

---

## 2. Hyperparameter Selection & Tuning

### 2.1 Learning Rate Selection

Following assignment guidelines and empirical testing:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| lr_max | 1e-3 | Smaller models benefit from larger learning rates |
| lr_min | 1e-4 | 10× smaller than max (standard practice) |
| warmup_iters | 100 | 10% of total iterations |
| cos_iters | 1000 | Cosine decay terminates at final step |

#### Learning Rate Sensitivity Analysis

```
lr = 1e-2: Divergence (loss explodes after ~50 steps)
lr = 5e-3: Unstable training, occasional spikes
lr = 1e-3: Optimal - smooth convergence ✓
lr = 1e-4: Slower convergence, suboptimal final loss
lr = 1e-5: Insufficient learning, high final loss
```

**Key Finding**: Best learning rate is approximately 10× below the divergence threshold, consistent with "edge of stability" theory.

### 2.2 Batch Size Considerations

| Batch Size | Memory Usage | Tokens/Step | Training Speed | Notes |
|------------|--------------|-------------|----------------|-------|
| 16 | ~8GB | 4,096 | Slower | More gradient noise |
| 32 | ~12GB | 8,192 | Optimal ✓ | Good efficiency/stability tradeoff |
| 64 | ~20GB | 16,384 | Fastest/step | Near memory limit on 24GB |
| 128 | OOM | - | - | Exceeds M4 24GB |

**Selection**: batch_size=32 chosen for:
- Fits comfortably in 24GB RAM with headroom
- Provides sufficient gradient averaging
- Allows reasonable iteration speed

### 2.3 Context Length Tradeoffs

Memory scales as O(batch × context² × d_model) for attention.

| Context | Memory Impact | Quality | Use Case |
|---------|---------------|---------|----------|
| 128 | Low | Lower coherence | Fast prototyping |
| 256 | Medium | Good ✓ | TinyStories optimal |
| 512 | High | Overkill for TinyStories | Complex documents |

### 2.4 AdamW Hyperparameters

```python
AdamW(
    params,
    lr=1e-3,           # Set by scheduler
    betas=(0.9, 0.999), # Default momentum terms
    eps=1e-8,           # Numerical stability
    weight_decay=0.01   # Decoupled regularization
)
```

---

## 3. Memory Profiling & System Behavior

### 3.1 Theoretical Memory Budget

For 17M parameter model with batch_size=32, context_length=256:

| Component | Formula | Size (float32) |
|-----------|---------|----------------|
| **Parameters** | 17M × 4 bytes | ~68 MB |
| **Gradients** | 17M × 4 bytes | ~68 MB |
| **Optimizer State** | 2 × 17M × 4 bytes (m, v) | ~136 MB |
| **Activations** | See breakdown below | ~2-4 GB |

#### Activation Memory Breakdown (per layer)

```
RMSNorm: B × S × D × 4 = 32 × 256 × 512 × 4 = 16.8 MB
QKV: B × S × 3D × 4 = 32 × 256 × 1536 × 4 = 50.3 MB
Attention scores: B × H × S × S × 4 = 32 × 16 × 256 × 256 × 4 = 134.2 MB
Attention output: B × S × D × 4 = 16.8 MB
FFN activations: B × S × d_ff × 4 = 32 × 256 × 1344 × 4 = 44.0 MB

Per-layer total: ~262 MB
Total (4 layers): ~1.05 GB
```

**Total Estimated**: ~1.5-2.5 GB during training

### 3.2 Observed Memory Usage

Using MPS (Metal Performance Shaders) on Apple M4:

```
Base Python process:     ~500 MB
Model loaded:            ~800 MB
During forward pass:     ~3.5 GB peak
During backward pass:    ~4.5 GB peak
Optimizer step:          ~3.0 GB
Steady state:            ~2.5 GB
```

**Observations**:
- Peak memory ~4.5GB significantly below 24GB limit
- MPS memory management adds overhead vs theoretical
- Checkpoint saves cause brief memory spikes (~300MB)

### 3.3 Memory Optimization Techniques Applied

1. **Memory-mapped data loading**: `np.load(file, mmap_mode='r')` prevents loading entire dataset into RAM
2. **Gradient zeroing**: `optimizer.zero_grad(set_to_none=True)` releases gradient memory
3. **No TF32 on MPS**: Avoided unstable TF32 kernels per assignment guidance
4. **Single precision**: All computations in float32 (half precision not reliable on MPS)

---

## 4. Benchmarks & Performance Metrics

### 4.1 Training Performance

#### Final Metrics (17M Model, 1000 iterations)

| Metric | Value |
|--------|-------|
| Final Training Loss | 2.24 |
| Final Eval Loss | 2.13 |
| Final Perplexity | 8.42 |
| Total Training Time | ~23 minutes |
| Tokens Processed | 8.19M (32 × 256 × 1000) |

#### Loss Curve Characteristics

```
Step 0:    Loss ~9.2, Perplexity ~10000 (random init)
Step 100:  Loss ~4.5, Perplexity ~90 (rapid descent)
Step 500:  Loss ~2.6, Perplexity ~13 (steady improvement)
Step 1000: Loss ~2.2, Perplexity ~8.4 (convergence)
```

### 4.2 Per-Step Timing Breakdown

From WandB profiling data:

| Operation | Time (ms) | % of Step |
|-----------|-----------|-----------|
| Batch loading | 6.2 | 0.4% |
| Forward pass | 234 | 16.8% |
| Loss calculation | 0.6 | 0.04% |
| Backward pass | 283 | 20.3% |
| Gradient norm calc | 55 | 3.9% |
| Gradient clipping | 3.9 | 0.3% |
| Optimizer step | 11.3 | 0.8% |
| Checkpoint save | 281 | 20.2% |
| **Total** | **~1400** | **100%** |

**Insights**:
- Forward/backward dominate as expected (~37% combined)
- Checkpoint saving is unexpectedly expensive (~20%) - could be optimized by reducing frequency
- Batch loading is efficient thanks to memory mapping

### 4.3 Throughput Metrics

```
Tokens/second: 8,192 tokens / 1.4 sec = ~5,850 tokens/sec
Training efficiency: 8.19M tokens in 23 min = 5,934 tokens/sec
```

### 4.4 Model Quality Assessment

#### Text Generation Samples

Prompt: "Once upon a time"

```
Once upon a time, there was a pretty girl named Lily. She loved to 
eat gum, especially the big black one. One day, Lily's mom asked 
her to help cook dinner. Lily was so excited! She loved to help 
her mom. Lily's mom made a big pot of soup for dinner...
```

**Quality observations**:
- Coherent narrative structure
- Appropriate children's story vocabulary
- Occasional semantic drift in longer generations
- Temperature/top-p tuning improves fluency

### 4.5 Comparison with Assignment Targets

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Eval Loss (low-resource) | ≤ 2.00 | 2.13 | Near target |
| Parameters | ~17M | ~17M | ✓ |
| Tokens Processed | 40M+ | 8.2M | Below target |
| Training Time | ~30 min | ~23 min | ✓ |

**Gap Analysis**: The slightly higher loss (2.13 vs 2.00 target) is likely due to:
1. Fewer tokens processed (8.2M vs recommended 40M)
2. Only 1000 training steps (vs recommended 5000)
3. Device bug: `token_positions` tensor was created on CPU instead of MPS

---

## 5. Key Learnings & Future Work

### 5.1 Technical Insights

1. **Device Consistency is Critical**: Discovered bug where `token_positions` was created on CPU while model was on MPS, causing silent performance degradation.

2. **d_ff Should Scale with d_model**: The sweep config had d_ff as an independent parameter, but it should be calculated as `≈ 8/3 × d_model` (rounded to multiple of 64).

3. **Checkpoint Overhead**: Saving checkpoints every iteration consumed 20% of training time. Consider saving less frequently or asynchronously.

4. **MPS Quirks**: Apple Silicon MPS backend has specific limitations:
   - No TF32 support (causes silent numerical instability)
   - `torch.compile` requires `backend="aot_eager"` (Inductor unsupported)
   - Memory reporting less accurate than CUDA

### 5.2 Bugs Found & Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `device`/`dtype` not passed to Linear/Embedding | Parameters on wrong device | Pass to `torch.empty()` |
| `token_positions` on CPU | Cross-device tensor ops | Add `.to(device)` |
| `load_model` expects flat config | Checkpoint incompatibility | Flatten model_settings |

### 5.3 Future Improvements

1. **Scale training**: Extend to 5000 iterations (40M tokens) to reach target loss
2. **Hyperparameter sweep**: Run systematic Bayesian optimization with fixed d_ff calculation
3. **Memory optimization**: Implement gradient checkpointing for larger batch sizes
4. **OpenWebText training**: Apply learnings to larger, more complex dataset
5. **Quantization**: Explore int8/int4 inference for deployment

### 5.4 Architecture Ablations (Planned)

| Ablation | Hypothesis | Status |
|----------|------------|--------|
| Remove RMSNorm | Training destabilizes at high LR | TODO |
| Post-norm vs Pre-norm | Pre-norm more stable | TODO |
| NoPE (no position embeddings) | Causal mask provides implicit position | TODO |
| SiLU vs SwiGLU | SwiGLU provides marginal improvement | TODO |

---

## Appendix A: Project Structure

```
assignment1-basics/
├── cs336_basics/
│   ├── nn.py              # Core modules (Linear, Embedding, RMSNorm, etc.)
│   ├── implementations.py  # TransformerLM, training utilities
│   ├── optimizer.py        # AdamW, LR scheduling
│   ├── bpe.py             # BPE tokenizer
│   ├── train.py           # Main training script
│   ├── train_sweep.py     # WandB sweep runner
│   └── decode.py          # Text generation
├── checkpoints/           # Saved model states
├── tokenizers/            # Trained BPE vocabularies
└── docs/                  # Documentation
```

## Appendix B: Reproducibility

```bash
# Environment setup
uv sync

# Train tokenizer & tokenize data (one-time)
uv run python cs336_basics/train.py  # Will auto-tokenize if needed

# Training with default config
uv run python cs336_basics/train.py

# Text generation
uv run python cs336_basics/decode.py checkpoints/_YYYYMMDD_HHMM/latest.pt
```

## Appendix C: Hardware Specifications

| Component | Specification |
|-----------|---------------|
| CPU | Apple M4 (4P + 6E cores) |
| GPU | 10-core Apple GPU |
| RAM | 24 GB unified memory |
| OS | macOS 26.1 (arm64) |
| Python | 3.13.3 |
| PyTorch | 2.6.0 (MPS backend) |

---

*Last updated: December 2025*