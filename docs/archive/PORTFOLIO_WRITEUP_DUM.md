# Portfolio Writeup — Training a Transformer Language Model from Scratch (CS336 Assignment 1 “Basics”)

## Project overview

This project is an end-to-end implementation of the core components needed to train a small decoder-only Transformer language model from scratch:

- **Tokenizer**: byte-level BPE training + serialization, plus encode/decode.
- **Model**: a **pre-norm Transformer** using **RoPE** self-attention and a **SwiGLU** feed-forward.
- **Optimization**: custom **AdamW** + cosine schedule with warmup.
- **Training**: a full training loop with tokenized dataset ingestion, checkpointing, evaluation, and experiment tracking (W&B support + local timing hooks).

The engineering goal was not just “get a model training,” but to make it **testable**, **resource-aware**, and **measurable**: dimensional analysis informed test configurations, memory expectations informed hyperparameter ranges, profiling guided optimization, and consistent metrics made regressions obvious.

## What I built (implementation highlights)

The core Transformer stack is implemented in `cs336_basics/implementations.py` using building blocks from `cs336_basics/nn.py` and a custom optimizer in `cs336_basics/optimizer.py`.

- **Transformer block**: pre-norm, RoPE MHA, SwiGLU FFN, two RMSNorms.
- **Tokenizer artifacts**: tokenized corpora stored as `.npy` arrays with dtype **uint16** (compact storage + memmap-friendly).
- **Training loop**: includes step-level timing hooks (batch load / forward / backward / optimizer step / checkpoint save) in `cs336_basics/train.py`.

## Dimensional analysis → resource accounting (and how it drove testing)

### Data representation sizes (tokenized corpora)

Token IDs are stored as `uint16` ($2$ bytes/token). This matches typical TinyStories vocab sizes (≤ 10k) and keeps files compact.

From the on-disk tokenized arrays in this repo:

- **Train**: `tokenized/tinystories_train.npy`
  - dtype `<u2` (uint16), shape `(560,259,947,)`
  - file size ≈ **1.12 GB**
- **Valid**: `tokenized/tinystories_valid.npy`
  - dtype `<u2`, shape `(5,659,362,)`
  - file size ≈ **11.3 MB**

Why this matters for testing:

- Loading tokens with `mmap_mode="r"` keeps RAM roughly constant while training on a large file.
- It makes it feasible to run training/eval loops on laptops without needing 1–2GB of extra RAM for the dataset in addition to model activations.

### Model parameter count (exact for *this* implementation)

This codebase’s `TransformerLM` (see `cs336_basics/implementations.py`) uses:

- Token embedding: `Embedding(vocab_size, d_model)`
- `num_layers` blocks of:
  - RoPE MHA with **four** dense matrices: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$
  - SwiGLU FFN with **three** dense matrices: $W_1, W_3 \in \mathbb{R}^{d_{ff} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{ff}}$
  - 2× RMSNorm (scale vector only)
- Final RMSNorm
- LM head `Linear(d_model, vocab_size)`

Let:

- $V$ = vocab_size
- $d$ = d_model
- $h$ = num_heads (note: doesn’t change parameter count here because projections are full $d\times d$)
- $L$ = num_layers
- $f$ = d_ff

Then:

- **Embeddings + LM head**: $2Vd$
- **Per-layer attention**: $4d^2$
- **Per-layer SwiGLU FFN**: $3df$
- **Per-layer RMSNorms**: $2d$
- **Final RMSNorm**: $d$

Total params:

$$
N_{\text{params}} = 2Vd + L(4d^2 + 3df + 2d) + d
$$

For the TinyStories “baseline-ish” config used in the assignment writeup:

- $V=10,000$
- $d=512$
- $f=1344$
- $L=4$

Compute:

- $2Vd = 2 \cdot 10,000 \cdot 512 = 10,240,000$
- $4d^2 = 4 \cdot 512^2 = 1,048,576$ per layer
- $3df = 3 \cdot 512 \cdot 1344 = 2,064,384$ per layer
- $2d = 1,024$ per layer
- Per layer total: $3,113,984$
- 4 layers: $12,455,936$
- Final RMSNorm: $+512$

So:

- **Total**: $10,240,000 + 12,455,936 + 512 = 22,696,448$ parameters (**~22.7M**)

Why this matters for testing:

- A 22.7M-param model is small enough to iterate locally but large enough that **activation memory dominates** (especially attention), so “it runs on CPU” does not imply “it won’t OOM on MPS at larger context.”
- It also makes it easy to create “unit-testable” downscaled configs that preserve shape logic (e.g., keep `d_model % num_heads == 0`) while drastically reducing memory.

### Optimizer + gradient memory (rule-of-thumb)

Training in float32 typically incurs (per parameter):

- weights: 4 bytes
- gradients: 4 bytes
- AdamW states $m, v$: 8 bytes total (two float32 tensors)

So a simple approximation is **~16 bytes/parameter** during training (not counting activation buffers):

$$
M_{\text{params+opt}} \approx 16 \cdot N_{\text{params}}
$$

For $N \approx 22.7$M:

- $16N \approx 363$ MB

This is *not* what usually blows up memory; activations do.

### Activation memory (why attention drives laptop-safe configs)

For a decoder-only Transformer, attention has an $O(S^2)$ term where $S$ is sequence length (`context_length`). The attention score/probability tensors have shape roughly:

$$
(B, h, S, S)
$$

So, at float32, attention alone costs approximately:

$$
M_{\text{attn}} \approx 4 \cdot B \cdot h \cdot S^2 \text{ bytes}
$$

Example (baseline-ish): $B=32$, $h=16$, $S=256$

- elements: $32 \cdot 16 \cdot 256^2 = 33,554,432$
- bytes: $33,554,432 \cdot 4 \approx 134$ MB **per layer** (just for one $B\times h\times S\times S$ tensor)

This scaling explains two practical decisions:

- **Why “just try context_length=512” is a big deal**: doubling $S$ quadruples $S^2$ ⇒ ~4× attention memory.
- **Why I downscale tests using smaller $S$**: most correctness bugs show up with $S=32$ or $S=64$, while memory blowups show up once $S$ gets large.

### Testing strategy driven by the math

I used a tiered strategy where early tests stress **correctness of shapes and numerics** with small dimensions, and later tests stress **system behavior** with realistic sizes:

- **Unit tests / invariants (fast)**
  - Verify per-module shape contracts (`Embedding`, `RMSNorm`, `scaled_dot_product_attention`, etc.).
  - Validate divisibility constraints (`d_model % num_heads == 0`) and mask behavior.
- **Single-minibatch overfit (debugging correctness)**
  - Downscale to something like $B=4, S=64, d=128, L=2, h=4$ and train until near-zero training loss.
  - This quickly exposes bugs in loss alignment, masking, or optimizer state.
- **Integration test (resource realism)**
  - Move to `context_length=256` and batch sizes that match memory expectations.
  - This is where attention memory and dataloading overhead become visible.

## Pretraining pipeline and hyperparameter tuning

### Pipeline stages

1. **Train BPE tokenizer** on TinyStories (`vocab_size=10,000`, include `<|endoftext|>`).
2. **Tokenize the corpus** into a contiguous uint16 `.npy` array.
3. **Train the Transformer LM** on the token array with periodic validation evaluation.
4. **Decode/generate text** from checkpoints to sanity-check learned fluency.

### Hyperparameter tuning approach (resource-aware)

The assignment guidance emphasizes keeping the architecture fixed (TinyStories baseline) and doing a **learning-rate sweep**.

On laptop-class machines (CPU/MPS), the assignment recommends reducing total tokens processed to ~40M, e.g.:

$$
\text{tokens} = B \cdot \text{steps} \cdot S \approx 32 \cdot 5000 \cdot 256 \approx 40.96\text{M}
$$

This directly informs a laptop-safe sweep plan:

- **Fix**: architecture (`d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `context_length=256`)
- **Fix**: token budget (~40M tokens on MPS/CPU)
- **Sweep**: learning rate (and optionally warmup + weight_decay)
- **Measure**: `eval_loss` and wallclock time-to-loss

This avoids the common pitfall where changing `batch_size` and `context_length` changes the token budget so much that runs aren’t comparable.

## Profiling & observed system behavior (vs expectations)

### Tokenizer training profiling (cProfile evidence)

This repo contains two cProfile outputs for BPE training:

- `tokenizing_artifacts/bpe_train_output1.prof`
  - ~**1252.9s** total (~20.9 minutes)
  - Dominant hotspot: `bpe.py:update_tokens` (~885s self time, ~1163s cumulative)
- `tokenizing_artifacts/bpe_train_output2.prof`
  - ~**1069.2s** total (~17.8 minutes)
  - Dominant hotspot: `bpe.py:update_tokens_v2` (~681.6s self time, ~970.9s cumulative)

Interpretation:

- The work is heavily dominated by a Python-level token update loop (lots of `len`, list `append`, and `max`), which matches the expectation that naïve BPE updates are algorithmically expensive.
- The second profile shows a meaningful speedup (~15% overall) after an update routine change—confirming that profiling-guided optimization was effective.
- Multiprocessing orchestration and IPC shows up as tens of seconds of overhead (pool termination / read / recv), which is expected for chunked parallelism.

### Training loop measurement hooks (timing breakdown)

The training loop in `cs336_basics/train.py` already includes timers for:

- batch construction
- forward pass
- loss computation
- backward pass
- gradient norm calculation + clipping
- optimizer step
- checkpoint saving

These timers support a benchmark narrative such as:

- “On laptop hardware, forward/backward dominates; batch construction is cheap due to memmap + vectorized indexing; checkpointing becomes visible if done too frequently.”

### Notes on inconsistencies found while reading

`tokenizing_artifacts/train_tiny_stories.py` sets:

- `output_path = "./tokenziers/tinystories/"`

That looks like a typo (`tokenziers` vs `tokenizers`) and would cause saved artifacts to land in an unexpected directory. This is the kind of “small paper-cut bug” that a portfolio writeup can mention as an example of catching integration issues early.

## Benchmarks & metrics (what I track and why)

### Tokenizer metrics

- **BPE training wallclock**: total time to reach vocab size $V$ (measured; profiled).
- **Tokenizer throughput**: bytes/sec or tokens/sec when encoding text streams.
- **Compression ratio** (bytes/token): aligns with assignment deliverables and connects to downstream memory footprint.

### Model/training metrics

- **Training loss** and **validation loss** (cross-entropy).
- **Perplexity**: $ \exp(\text{loss}) $ for interpretability.
- **Tokens/sec**: $B \cdot S / \text{seconds per step}$ (primary throughput metric).
- **Time-to-quality**: wallclock to reach a target validation loss (e.g., ≤ 2.0 on the 40M-token low-resource run).
- **Stability diagnostics**:
  - gradient norms (pre/post clip)
  - divergence detection (loss spikes, NaNs)

### System metrics (for memory + runtime characterization)

- **Peak resident memory (RSS)** over time (CPU-side).
- **MPS allocated memory** (if applicable) to correlate with predicted attention scaling.
- **Step time breakdown** from the training loop timers to identify bottlenecks (data vs compute vs checkpoint I/O).

## What I would put in the “Results” section (template you can fill per machine)

| Category | Metric | Value | Notes |
|---|---:|---:|---|
| Data | Train tokens | 560,259,947 | `uint16`, memmap |
| Data | Valid tokens | 5,659,362 | `uint16`, memmap |
| Tokenizer | BPE train time (profile 1) | 1252.9s | hotspot: `update_tokens` |
| Tokenizer | BPE train time (profile 2) | 1069.2s | hotspot: `update_tokens_v2` |
| Model | Params (baseline-ish) | 22,696,448 | no weight tying |
| Train | Step time (median) | _(fill)_ | forward/backward breakdown |
| Train | Tokens/sec | _(fill)_ | compare CPU vs MPS |
| Eval | Best val loss @ step X | _(fill)_ | report step + wallclock |
| Eval | Best val ppl | _(fill)_ | $e^{\text{loss}}$ |

## MacBook Air M4 (24GB) — recommended “runs clean” settings

For a MacBook Air–class Apple Silicon machine, the safest approach is:

- **Keep `context_length=256` for most work**; treat `512` as “large experiment” due to $S^2$ attention memory.
- Start with **batch size 16 or 32**; increase only if memory headroom remains stable.
- Use the assignment’s low-resource token budget target (~40M tokens) and keep token budget fixed across runs.
- Avoid changes that inflate attention tensors (simultaneously increasing `batch_size`, `context_length`, and `num_layers`).

If you want a single “portfolio demo” run that’s likely to complete reliably:

- $d=512, L=4, h=16, f=1344, S=256$
- $B=32, \text{steps}=5000$ (≈ 41M tokens)
- LR sweep over a small set (e.g., 5–10 values) while holding everything else fixed.

## Takeaways

- **Dimensional analysis** (especially $S^2$ attention scaling) is the difference between “it sometimes OOMs” and “it reliably runs.”
- **Profiling** validated expectations: the slowest components were Python-heavy BPE updates and attention-heavy training at larger contexts.
- **Benchmarks** made iteration efficient: step timing + tokens/sec + val loss curves are enough to compare code changes and hyperparameters systematically.


