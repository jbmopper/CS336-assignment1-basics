import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # local_tiny.py
    ## benchmarking/sizing/traning the net on tinystories, locally

    1. lay out the architecture and identify points of interest
    2. understand the M4 system
        2. notes from assignment:
            - Do NOT use torch.set_float32_matmul_precision('high') - causes silent bugs
            - Can use torch.compile(model, backend="aot_eager") for modest speedup
            - Adjust cosine LR schedule to reach minimum at exactly step X
    3. benchmark different parts
        1. get a feel for what bad/good sizes are
        2. practice using benchmark
    4. run some sweeps
        1. assignment suggests sweeping over learing rate to identify where things get unstable
            1. target validation loss of ≤2.00
        3. test different batch sizes until OOM (fun!)
    5. try some ablations
        1. ablate RoPE (version already exists)
        2. ablate all RMSnorms and test previous optmial learning rate
        3. try post-norm instead of pre-norm (may skip)
        4. try using SiLU instead of SwiGLU
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notes on the system (generated)

    ## M4 MacBook Air Hardware Specs

    Your 24GB M4 MacBook Air has:
    - **10-core GPU** (8 performance + 2 efficiency, or 10-core variant)
    - **120 GB/s memory bandwidth** (notably lower than M4 Pro's 273GB/s or NVIDIA GPUs' 1TB/s+)
    - **16-core Neural Engine** (38 TOPS)
    - **Unified Memory Architecture** - CPU and GPU share the same physical memory pool

    ## Key Optimization Insights

    ### 1. Memory Architecture (Unified Memory)

    The unified memory is both a strength and constraint:

    **Advantages:**
    - Zero-copy tensor transfers between CPU/GPU - no PCIe bottleneck
    - Your full 24GB is available to both CPU and GPU (vs. discrete GPUs with separate VRAM)
    - Great for prototyping and fitting larger models than would fit in typical GPU VRAM

    **Constraints:**
    - **120 GB/s bandwidth is ~8-10x lower than datacenter GPUs** - this is the main bottleneck for large matrix multiplications
    - **Individual tensor operations are limited to 2^32 bytes (4GB)** - this is a Metal/MPS limitation, not a RAM limitation. Attention matrices with long sequences can hit this.

    ### 2. Optimal Dimensions for MPS

    Based on the research, here are recommended guidelines:

    | Parameter | Recommendation | Rationale |
    |-----------|---------------|-----------|
    | **d_model** | 256, 512, 768, or 1024 | Multiples of 64/128 align well with GPU warp sizes |
    | **d_ff** | 4× d_model (or ~2.67× with SwiGLU) | Standard transformer ratio |
    | **num_heads** | d_model / 64 or d_model / 128 | Head dim of 64-128 is typical |
    | **Context length** | ≤2048 initially | Attention is O(n²); longer sequences risk hitting 4GB tensor limit |
    | **Batch size** | 16-64 for training | Memory scales with batch × context² × d_model |

    One note: `num_heads=16` with `d_model=512` gives a head dimension of 32, which is smaller than typical (64-128). You might see slight efficiency gains with `num_heads=8` (head_dim=64).

    ### 3. MPS-Specific Optimizations

    **Precision:**
    - MPS prefers **float32** over float16 - FP16 can cause numerical instability in softmax, especially with long sequences
    - MPS doesn't support FP8/FP4 or FlashAttention natively

    **Attention for Long Sequences:**
    If you extend context_length beyond ~4K tokens, implement **attention chunking/slicing**:
    ```python
    # Instead of computing full attention matrix at once,
    # process in chunks to stay under 4GB tensor limit
    ```

    ### 4. Practical Batch Size Selection

    Memory scales roughly as: `batch_size × context_length² × d_model × 4 bytes`

    For your config (d_model=512, context_length=256):
    - **batch_size=32**: ~134 MB for attention matrices alone
    - **batch_size=64**: ~268 MB
    - **batch_size=128**: ~536 MB

    With 24GB unified memory and ~17M params (~68MB for model), you have headroom. Try:
    - Start with batch_size=64-128 for training
    - Use gradient accumulation if you want effective batch sizes of 256+

    ### 5. Bandwidth-Bound Optimizations

    Since M4's 120GB/s bandwidth is the main bottleneck:

    1. **Reduce memory traffic** - fuse operations where possible
    2. **Avoid CPU fallbacks** - check for MPS-unsupported ops
    3. **Use memory-mapped data loading** (you're already doing this with `mmap_mode='r'`)
    4. **Consider MLX** for inference-heavy workloads - it's specifically optimized for Apple Silicon

    ### Summary Recommendations for Your Setup

    | Aspect | Current | Suggested |
    |--------|---------|-----------|
    | d_model | 512 | Good - keep |
    | num_heads | 16 | Try 8 (head_dim=64) |
    | d_ff | 1344 | Could try 2048 (4×512) |
    | context_length | 256 | Good for training; can push to 512-1024 |
    | batch_size | 32 | Try 64-128 |
    | dtype | (default float32) | Keep float32 for MPS stability |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's start by initializing the config dict and loading the data
    """)
    return


@app.cell
def _(np):
    config = {}

    # files exist
    config["train_file"] = "tokenized/tinystories_train_fixed.npy"
    config["valid_file"] = "tokenized/tinystories_valid_fixed.npy"

    # Load tokenized data
    tokens = np.load(config["train_file"], mmap_mode='r')
    print(f"Training tokens loaded from {config['train_file']}, shape {tokens.shape}.")
    
    valid_tokens = np.load(config["valid_file"], mmap_mode='r')
    print(f"Validation tokens loaded from {config['valid_file']}, shape {valid_tokens.shape}.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we can review the training class we've made:
    """)
    return


if __name__ == "__main__":
    app.run()
