import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


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

    Your MacBook Air has:
    - **10-core CPU** (4 performance + 6 efficiency)
    - **10-core GPU** (uniform Apple GPU cores, Metal/MPS backend)
    - **24 GB RAM** (Unified Memory)
    - **~120 GB/s memory bandwidth** (notably lower than M4 Pro's ~273 GB/s or NVIDIA GPUs' 1 TB/s+)
    - **16-core Neural Engine** (38 TOPS)
    - **Unified Memory Architecture** – CPU and GPU share the same physical memory pool

    ## Key Optimization Insights

    ### 1. Memory Architecture (Unified Memory)

    The unified memory model is both a strength and a constraint.

    **Advantages:**
    - Zero-copy tensor sharing between CPU and GPU (no PCIe transfer overhead)
    - The full 24 GB is available to both CPU and GPU (unlike discrete GPUs with separate VRAM)
    - Enables fitting larger models than typical consumer GPU VRAM limits, useful for prototyping

    **Constraints:**
    - **Memory bandwidth is the main bottleneck.** At ~120 GB/s, it is ~8–10× lower than datacenter GPUs and limits large matrix multiplication throughput.
    - Some MPS kernels use **32-bit indexing** and can fail on very large tensors (≈4 GB+), especially attention matrices at long sequence lengths.
      This is an implementation constraint, not a fundamental limit of unified memory itself.
    - Unified memory makes large models possible, but not fast; most transformer workloads remain bandwidth-bound.

    ---

    ### 2. Optimal Dimensions for MPS

    Recommended architectural guidelines:

    | Parameter | Recommendation | Rationale |
    |--------|---------------|---------|
    | **d_model** | 256, 512, 768, or 1024 | Multiples of 64/128 align well with Apple GPU vectorization and memory tiling |
    | **d_ff** | 4× d_model (or ~2.67× with SwiGLU) | Standard transformer ratio |
    | **num_heads** | d_model / 64 or d_model / 128 | Head dim of 64–128 is typical |
    | **Context length** | ≤2048 initially | Attention is O(n²); longer sequences increase memory pressure |
    | **Batch size** | 16–64 initially | Scales quadratically with context length |

    Note:
    `num_heads=16` with `d_model=512` gives a head dimension of 32, which is smaller than typical (64–128).
    You may see better efficiency with `num_heads=8` (head_dim=64).

    ---

    ### 3. MPS-Specific Optimizations

    **Precision:**
    - Float32 is often more numerically stable on MPS for training.
    - Float16 can work but may cause instability in softmax and attention, especially for long sequences.
    - MPS does not support FP8/FP4 or FlashAttention natively.

    **Attention for Long Sequences:**
    If extending `context_length` beyond a few thousand tokens, use **attention chunking/slicing**:

    ```python
    # Instead of computing the full attention matrix at once,
    # process it in blocks to avoid very large intermediate tensors
    ```

    ---

    ### 4. Practical Batch Size Selection

    Attention memory scales approximately as:

    ```
    batch_size × num_heads × context_length² × bytes_per_element
    ```

    For example, with:

    * `d_model = 512`
    * `num_heads = 16`
    * `context_length = 256`
    * `float32` (4 bytes)

    Approximate attention memory:

    * **batch_size = 32** →
      32 × 16 × 256² × 4 bytes ≈ **134 MB**
    * **batch_size = 64** →
      ≈ **268 MB**
    * **batch_size = 128** →
      ≈ **536 MB**

    This is only for attention matrices; activations, gradients, optimizer state, and parameters add additional overhead.

    With 24 GB unified memory and ~17M parameters (~68 MB for weights in float32), you have significant headroom, but bandwidth will limit performance before capacity does.

    Practical guidance:

    * Start with `batch_size = 64–128`
    * Increase until throughput stops scaling or memory pressure appears
    * Use gradient accumulation to reach larger effective batch sizes

    ---

    ### 5. Bandwidth-Bound Optimizations

    Since ~120 GB/s bandwidth is the main constraint:

    1. **Reduce memory traffic** – fuse operations where possible
    2. **Avoid CPU fallbacks** – ensure ops are supported on MPS
    3. **Use memory-mapped loading** (`mmap_mode='r'`) to avoid unnecessary copies
    4. **Consider MLX** for inference-heavy workloads; it is tuned specifically for Apple Silicon

    ---

    ### Summary Recommendations

    | Aspect         | Assignment Default | Suggested                      |
    | -------------- | ------- | ----------------------------------------- |
    | d_model        | 512     | Good for clean head factorization – keep                               |
    | num_heads      | 16      | Try 8 (head_dim=64)                       |
    | d_ff           | 1344 (~8/3 × d_model, multiple of 64)   | Consider 2048 (4×512)                     |
    | context_length | 256     | Good for training; push to 512–1024 later |
    | batch_size     | 32      | Try 64–128                                |
    | dtype          | float32 | Keep for MPS stability

    **Staff reference (M3 Max, 36GB):**
    - Config: batch=32 × steps=5000 × context=256 = 40.96M tokens
    - Time: ~36 min on MPS
    - Result: val loss 1.80 at step 5000
    - Target: val loss ≤2.00
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Architecture
    """)
    return


@app.cell
def _(mo):
    # Primary model hyperparameters (user inputs)
    B = mo.ui.slider(1, 128, step=1, value=32, label="B (batch size)")
    seq_len = mo.ui.slider(128, 4096, step=128, value=512, label="seq_len (context length)")
    V = mo.ui.slider(1000, 100000, step=1000, value=50257, label="V (vocab size)")
    d_model = mo.ui.slider(128, 2048, step=64, value=512, label="d_model")
    n_heads = mo.ui.slider(1, 32, step=1, value=8, label="h (num_heads)")
    n_blocks = mo.ui.slider(1, 48, step=1, value=12, label="n_blocks (layers)")
    d_ff = mo.ui.slider(64, 6400, step=64, value=1344, label="d_ff (feed-forward dimension)")

    # Data types
    wt_dtype = mo.ui.dropdown(["float32", "float16", "bfloat16", "float8"], value="float32", label="wt_dtype (weights)")
    ft_dtype = mo.ui.dropdown(["float32", "float16", "bfloat16", "float8"], value="float32", label="ft_dtype (features)")

    controls = mo.vstack([
        mo.md("### Parameters and Adjustment"),
        mo.hstack([B, seq_len, V]),
        mo.hstack([d_model, n_heads, n_blocks]),
        mo.hstack([d_ff, wt_dtype, ft_dtype]),
    ])
    controls
    return B, V, d_ff, d_model, ft_dtype, n_blocks, n_heads, seq_len, wt_dtype


@app.cell
def _(B, V, d_ff, d_model, ft_dtype, n_blocks, n_heads, seq_len, wt_dtype):
    # Derived values
    d_head = d_model.value // n_heads.value
    # d_ff = d_ff.value
    _3d_model = 3 * d_model.value

    # Bytes per element
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "float8": 1}
    wt_bytes = dtype_bytes[wt_dtype.value]
    ft_bytes = dtype_bytes[ft_dtype.value]

    # Size calculations
    input_size = B.value * seq_len.value * 2  # int16
    emb_size = V.value * d_model.value
    ft_size = B.value * seq_len.value * d_model.value * ft_bytes
    RMS_size = d_model.value * wt_bytes
    wqkv_size = d_model.value * _3d_model * wt_bytes
    qkv_size = B.value * seq_len.value * _3d_model * ft_bytes
    head_size = B.value * n_heads.value * seq_len.value * d_head * ft_bytes
    features_size = B.value * seq_len.value * d_model.value * ft_bytes
    swiglu_size = 3 * d_model.value * d_ff.value * wt_bytes
    lm_head_size = V.value * d_model.value * wt_bytes
    output_size = B.value * seq_len.value * V.value * ft_bytes
    o_size = d_model.value * d_model.value * wt_bytes

    # total model size
    per_block_size = RMS_size + wqkv_size + o_size + RMS_size + swiglu_size
    total_weights = emb_size + per_block_size * n_blocks.value + RMS_size + head_size

    # Compute estimates (FLOPs)
    rms_norm_comp = 2 * B.value * seq_len.value * d_model.value
    QKV_comp = 2 * B.value * seq_len.value * d_model.value * _3d_model
    RoPE_comp = 2 * B.value * n_heads.value * seq_len.value * d_head
    QK_compute = 2 * B.value * n_heads.value * seq_len.value * seq_len.value * d_head
    softmax_compute = 3 * B.value * n_heads.value * seq_len.value * seq_len.value
    SDPA_compute = QK_compute + softmax_compute + 2 * B.value * n_heads.value * seq_len.value * seq_len.value * d_head
    swiglu_comp = 2 * B.value * seq_len.value * d_model.value * d_ff.value * 3
    lm_comp = 2 * B.value * seq_len.value * d_model.value * V.value
    o_proj_comp = B.value * seq_len.value * d_model.value * d_model.value

    # Total forward pass: per-block ops * n_blocks + final rms_norm + lm_head
    per_block_comp = 2*rms_norm_comp + QKV_comp + 2*RoPE_comp + SDPA_compute + o_proj_comp + swiglu_comp
    total_forward = n_blocks.value * per_block_comp + rms_norm_comp + lm_comp

    def fmt_size(b):
        if b >= 1e9: return f"{b/1e9:.2f} GB"
        if b >= 1e6: return f"{b/1e6:.2f} MB"
        if b >= 1e3: return f"{b/1e3:.2f} KB"
        return f"{b} B"

    def fmt_flops(f):
        if f >= 1e12: return f"{f/1e12:.2f} TFLOPs"
        if f >= 1e9: return f"{f/1e9:.2f} GFLOPs"
        if f >= 1e6: return f"{f/1e6:.2f} MFLOPs"
        return f"{f:.0f} FLOPs"

    # Return all values needed for SVG substitution
    svg_vars = {
        "B": B.value,
        "seq_len": seq_len.value,
        "V": V.value,
        "d_model": d_model.value,
        "h": n_heads.value,
        "n_blocks": n_blocks.value,
        "d_ff": d_ff.value,
        "d_head": d_head,
        "3d_model": _3d_model,
        "wt_dtype": wt_dtype.value,
        "ft_dtype": ft_dtype.value,
        # Sizes (formatted)
        "input_size": fmt_size(input_size),
        "emb_size": fmt_size(emb_size),
        "ft_size": fmt_size(ft_size),
        "RMS_size": fmt_size(RMS_size),
        "wqkv_size": fmt_size(wqkv_size),
        "qkv_size": fmt_size(qkv_size),
        "head_size": fmt_size(head_size),
        "features_size": fmt_size(features_size),
        "swiglu_size": fmt_size(swiglu_size),
        "lm_head_size": fmt_size(lm_head_size),
        "o_size": fmt_size(o_size),
        "output_size": fmt_size(output_size),
        "total_weights": fmt_size(total_weights),
        # Compute (formatted)
        "rms_norm_comp": fmt_flops(rms_norm_comp),
        "QKV_comp": fmt_flops(QKV_comp),
        "RoPE_comp": fmt_flops(RoPE_comp),
        "QK_compute": fmt_flops(QK_compute),
        "softmax_compute": fmt_flops(softmax_compute),
        "SDPA_compute": fmt_flops(SDPA_compute),
        "o_proj_comp": fmt_flops(o_proj_comp),
        "swiglu_comp": fmt_flops(swiglu_comp),
        "lm_comp": fmt_flops(lm_comp),
        "total_forward": fmt_flops(total_forward),
    }
    return (svg_vars,)


@app.cell
def _(mo, svg_vars):
    with open("notebooks/cs336_forward.svg") as f:
        svg = f.read()

    # Substitute all template variables from svg_vars
    for key, value in svg_vars.items():
        svg = svg.replace(f"{{{{{key}}}}}", str(value))

    # inject font styling
    font_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    text {
      font-family: "Inter", system-ui, -apple-system, sans-serif;
    }
    </style>
    """

    svg = svg.replace(">", f">{font_style}", 1)

    mo.Html(svg)
    return


@app.cell
def _(mo, svg_vars):
    mo.md(f"""
    ### Computed Values

    | Variable | Value |
    |----------|-------|
    | d_head | {svg_vars['d_head']} |
    | 3·d_model | {svg_vars['3d_model']} |

    ### Tensor Sizes

    | Tensor | Size |
    |--------|------|
    | input_size (int16) | {svg_vars['input_size']} |
    | ft_size [B,S,d_model] | {svg_vars['ft_size']} |
    | RMS weights | {svg_vars['RMS_size']} |
    | WQKV [d_model, 3·d_model] | {svg_vars['wqkv_size']} |
    | QKV [B,S,3·d_model] | {svg_vars['qkv_size']} |
    | head [B,h,S,d_head] | {svg_vars['head_size']} |
    | O proj [d_model, d_model] | {svg_vars['o_size']} |
    | features [B,S,d_model] | {svg_vars['features_size']} |
    | swiglu weights | {svg_vars['swiglu_size']} |
    | lm_head [V,d_model] | {svg_vars['lm_head_size']} |
    | output [B,S,V] | {svg_vars['output_size']} |

    ### Compute per Block

    | Operation | FLOPs |
    |-----------|-------|
    | rms_norm_comp | {svg_vars['rms_norm_comp']} |
    | QKV_comp | {svg_vars['QKV_comp']} |
    | RoPE_comp | {svg_vars['RoPE_comp']} |
    | SDPA_compute | {svg_vars['SDPA_compute']} |
    | o_proj_comp | {svg_vars['o_proj_comp']} |
    | swiglu_comp | {svg_vars['swiglu_comp']} |
    | lm_comp | {svg_vars['lm_comp']} |

    #### Forward Pass Totals
    **Compute:** {svg_vars['total_forward']}  
    **Weights:** {svg_vars['total_weights']}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Benchmarks

    Next we can review the training class we've made:
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


if __name__ == "__main__":
    app.run()
