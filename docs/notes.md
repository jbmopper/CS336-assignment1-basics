## Model selection
### (M4 traning)

To explore the properties of models with similar training characteristics but different architectures, two models that were closely matched for token throughput and memory consumption were selected:

Model A:
- batch_size: 64
- seq_len: 256
- d_model: 640
- d_head: 64
- num_heads: 10
- num_layers: 10
- d_ff: 1024

Model B:
- batch_size: 48
- seq_len: 256
- d_model: 384
- d_head: 32
- num_heads: 12
- num_layers: 12
- d_ff: 1728

Both of these used about 16GB of RAM and were observed to have 3800-3900 tokens/s throughput.

---

## Machine Opinion

### Analysis of Original Selection

The original Model A vs Model B comparison has a confound: Model A has both a low FFN ratio (1.6×) AND fewer layers (10 vs 12). This makes it difficult to attribute performance differences to a single architectural variable.

| Aspect | Model A | Model B |
|--------|---------|---------|
| d_model | 640 | 384 |
| num_layers | 10 | 12 |
| d_ff | 1024 | 1728 |
| **FFN ratio** | **1.6×** | **4.5×** |
| **Parameters** | **~48.9M** | **~38.7M** |

Model A has ~27% more parameters but a severely compressed FFN ratio. FFN layers act as key-value memories for factual knowledge, so Model A may underperform despite its parameter advantage.

### Recommended Alternative Selection

For a cleaner ablation isolating FFN ratio, consider these matched configs from the benchmark data (~4300 tok/s, ~15GB, ~36M params):

**Standard FFN Model**:
- batch_size: 48
- seq_len: 256
- d_model: 384
- d_head: 64
- num_heads: 6
- num_layers: 10
- d_ff: 2048
- FFN ratio: 5.3×
- Parameters: ~37.2M
- Throughput: ~4323 tok/s
- Memory: ~14.8 GB

**Wide-Attention Model**:
- batch_size: 64
- seq_len: 256
- d_model: 512
- d_head: 64
- num_heads: 8
- num_layers: 12
- d_ff: 704
- FFN ratio: 1.4×
- Parameters: ~35.8M
- Throughput: ~4377 tok/s
- Memory: ~15.1 GB

### Hypothesis

The Standard FFN model should achieve lower perplexity despite similar training cost, because:
1. FFN layers store learned factual knowledge
2. Standard 4× expansion is well-validated in literature
3. The wide-attention model sacrifices knowledge capacity for more attention heads

### Key Variables Worth Studying

| Variable | Question |
|----------|----------|
| FFN ratio | Knowledge capacity vs attention capacity |
| Depth vs Width | num_layers vs d_model scaling |
| Head dimension | d_head=32 vs 64 vs 128 efficiency |
| Sequence length | Context window vs compute trade-off |

---

## Machine Opinion 2

If your goal is “understand behavior” on an M4 without confounding by wall-clock, I’d pick a **near-perfectly cost-matched** pair and then interpret the results as a **width–depth tradeoff** (with `d_head` held fixed).

### My recommended comparison pair (from `train_benchmark_20260127_092323.json`)

**Model C (narrower, deeper):**
- batch_size: 48
- seq_len: 256
- d_model: 640
- d_head: 128
- num_heads: 5
- num_layers: 12
- d_ff: 1728  (FFN ratio \(\\approx 2.70\\times\))
- Bench: ~2646 tok/s, ~13.22 GB est.

**Model D (wider, shallower):**
- batch_size: 48
- seq_len: 256
- d_model: 768
- d_head: 128
- num_heads: 6
- num_layers: 10
- d_ff: 2048  (FFN ratio \(\\approx 2.67\\times\))
- Bench: ~2652 tok/s, ~13.27 GB est.

### Why these two
- **Fair wall-clock**: they’re essentially tied on tok/s and memory in your benchmark sweep, so you’re not comparing “more compute” vs “less compute”.
- **Isolated variable**: the main difference is **width vs depth** (`d_model` up, `num_layers` down) while keeping `B`, `S`, and `d_head` aligned.
- **Reasonable FFN capacity**: both are in the “normal-ish” SwiGLU ratio neighborhood (\(d_{ff}/d \\approx 2.7\\)).

### What I’d watch for (to build intuition)
- **Early optimization**: does one reach a given loss faster in steps (even if wall-clock is matched)?
- **Stability**: does the wider model tolerate higher LR / train more smoothly?
- **Generalization**: after equal tokens, which has lower val loss? (Scaling laws often like width; some regimes like depth.)

### Caveat
Your benchmark file often records `num_runs=1` because blocks can satisfy `min_run_time` in one block. Treat “tok/s match” as “very likely close”, but don’t over-index on <5% differences without a quick re-run.

---

## Machine Opinion 3

### The Core Question: Depth (Layers) vs. Knowledge Capacity (FFN Width)

On the M4 architecture, where throughput is heavily constrained by the ~120 GB/s memory bandwidth, every architectural choice is a trade-off in shuttling data between Unified Memory and GPU cores. The most interesting comparison is how to allocate this "bandwidth budget" between reasoning depth and factual storage.

### Recommended Comparison Pair

I have selected two models from the `train_benchmark_20260127_092323.json` sweep that have **nearly identical training throughput** (~3,850 tokens/sec) but represent opposite architectural philosophies. Both use `batch_size=32`, `seq_len=256`, and `d_model=512`.

| Parameter | **Setting A: "The Vertical Thinker"** | **Setting B: "The Horizontal Thinker"** |
| :--- | :--- | :--- |
| **Philosophy** | **Deep & Lean FFN** | **Shallow & Wide FFN** |
| **num_layers** | **14** | **10** |
| **d_ff** | **640** (1.25x expansion) | **1728** (~3.4x expansion) |
| **d_head** | 64 | 64 |
| **Throughput** | **3,874.6 tok/s** | **3,850.4 tok/s** |
| **Memory** | 8.82 GB | 8.24 GB |

### Why these two?

1.  **The "Equal Cost" Comparison**: Their throughput is almost identical, allowing for a fair wall-clock comparison. Any difference in final validation loss reflects architectural efficiency rather than compute investment.
2.  **Reasoning vs. Memorization**: Setting A tests whether the model benefits more from **14 stages of sequential transformation** (likely better for complex logic/grammar). Setting B tests whether it benefits more from **larger internal "fact" storage** in the FFN (likely better for memorizing TinyStories-style vocabulary).
3.  **Isolating the FFN Tax**: This pair allows you to see exactly how much depth you "pay" for a wider FFN while staying in the same performance bucket.

### Hypothesis for TinyStories

For a dataset like TinyStories, which features simple grammar but specific vocabulary and character facts, **Setting B** (the wide FFN) may outperform Setting A. The "reasoning" required for these stories is often shallower than the need to store associations and common patterns.
