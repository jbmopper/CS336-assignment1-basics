import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from cs336_basics.nn import scaled_dot_product_attention
    import torch.utils.benchmark as benchmark
    import torch
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
    import math as _math

    # Primary model hyperparameters (user inputs)
    B = mo.ui.slider(1, 128, step=1, value=32, label="B (batch size)", show_value=True)
    seq_len = mo.ui.slider(128, 4096, step=128, value=512, label="seq_len (context length)", show_value=True)
    V = mo.ui.slider(1000, 100000, step=1000, value=10000, label="V (vocab size)", show_value=True)
    d_model = mo.ui.slider(128, 2048, step=64, value=512, label="d_model", show_value=True)
    d_head = mo.ui.slider(32, 128, step=32, value=64, label="d_head", show_value=True)
    n_blocks = mo.ui.slider(1, 48, step=1, value=12, label="n_blocks (layers)", show_value=True)
    d_ff_slider = mo.ui.slider(64, 6400, step=64, value=1344, label="d_ff (manual)", show_value=True)

    # Toggle for locking d_ff to 8/3 * d_model (SwiGLU standard ratio)
    lock_d_ff = mo.ui.checkbox(value=True, label="Lock d_ff = ⌈8/3 × d_model⌉₆₄")

    # Data types
    wt_dtype = mo.ui.dropdown(["float32", "float16", "bfloat16", "float8"], value="float32", label="wt_dtype (weights)")
    ft_dtype = mo.ui.dropdown(["float32", "float16", "bfloat16", "float8"], value="float32", label="ft_dtype (features)")

    # Memory budget and training
    max_ram_gb = mo.ui.slider(4, 48, step=1, value=20, label="RAM Budget (GB)", show_value=True)
    train_steps = mo.ui.slider(100, 20000, step=100, value=5000, label="Training steps", show_value=True)

    controls = mo.vstack([
        mo.md("### Model Parameters"),
        mo.hstack([B, seq_len, V], justify="start", gap=2),
        mo.hstack([d_model, d_head, n_blocks], justify="start", gap=2),
        mo.hstack([lock_d_ff, d_ff_slider, wt_dtype, ft_dtype], justify="start", gap=2),
        mo.hstack([max_ram_gb, train_steps], justify="start", gap=2),
    ])
    controls
    return (
        B,
        V,
        d_ff_slider,
        d_head,
        d_model,
        ft_dtype,
        lock_d_ff,
        max_ram_gb,
        n_blocks,
        seq_len,
        train_steps,
        wt_dtype,
    )


@app.cell
def _(d_ff_slider, d_model, lock_d_ff):
    import math as _math

    # Compute d_ff: either locked to 8/3 * d_model (rounded up to multiple of 64) or manual
    if lock_d_ff.value:
        # SwiGLU standard: d_ff ≈ 8/3 * d_model, rounded up to nearest multiple of 64
        _raw_d_ff = d_model.value * 8 / 3
        d_ff = _math.ceil(_raw_d_ff / 64) * 64
    else:
        d_ff = d_ff_slider.value
    return (d_ff,)


@app.cell
def _(d_head, d_model):
    # Compute n_heads from d_model and d_head
    n_heads = d_model.value // d_head.value
    _remainder = d_model.value % d_head.value
    n_heads_valid = _remainder == 0
    return (n_heads, n_heads_valid)


@app.cell
def _(d_ff, d_head, d_model, lock_d_ff, mo, n_heads, n_heads_valid):
    _lock_status = f"**locked** = ⌈8/3 × {d_model.value}⌉₆₄ = {d_ff}" if lock_d_ff.value else f"manual = {d_ff}"
    _valid_status = "" if n_heads_valid else " ⚠️ **d_model not divisible by d_head!**"
    mo.md(f"""
    **Derived:** n_heads = {d_model.value} / {d_head.value} = {n_heads}{_valid_status}, d_ff ({_lock_status})
    """)
    return


@app.cell
def _(B, V, d_ff, d_head, d_model, ft_dtype, n_blocks, n_heads, seq_len, wt_dtype):
    # d_head is now a slider input, n_heads is computed
    _d_head = d_head.value
    _3d_model = 3 * d_model.value

    # Bytes per element
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "float8": 1}
    wt_bytes = dtype_bytes[wt_dtype.value]
    ft_bytes = dtype_bytes[ft_dtype.value]

    # Size calculations
    input_size = B.value * seq_len.value * 2  # int16
    emb_size = V.value * d_model.value * wt_bytes
    ft_size = B.value * seq_len.value * d_model.value * ft_bytes
    RMS_size = d_model.value * wt_bytes
    wqkv_size = d_model.value * _3d_model * wt_bytes
    qkv_size = B.value * seq_len.value * _3d_model * ft_bytes
    head_size = B.value * n_heads * seq_len.value * _d_head * ft_bytes  # n_heads is computed int
    features_size = B.value * seq_len.value * d_model.value * ft_bytes
    sp_size = B.value * n_heads * seq_len.value ** 2 * ft_bytes  # n_heads is computed int
    swiglu_size = 3 * d_model.value * d_ff * wt_bytes  # d_ff is now a computed int
    lm_head_size = V.value * d_model.value * wt_bytes
    output_size = B.value * seq_len.value * V.value * ft_bytes
    o_size = d_model.value * d_model.value * wt_bytes

    # total model size (weights)
    per_block_size = RMS_size + wqkv_size + o_size + RMS_size + swiglu_size
    total_weights = emb_size + per_block_size * n_blocks.value + RMS_size + lm_head_size

    # Compute estimates (FLOPs)
    rms_norm_comp = 2 * B.value * seq_len.value * d_model.value
    QKV_comp = 2 * B.value * seq_len.value * d_model.value * _3d_model
    RoPE_comp = 2 * B.value * n_heads * seq_len.value * _d_head  # n_heads is computed int
    QK_compute = 2 * B.value * n_heads * seq_len.value * seq_len.value * _d_head  # n_heads is computed int
    softmax_compute = 3 * B.value * n_heads * seq_len.value * seq_len.value  # n_heads is computed int
    SDPA_compute = QK_compute + softmax_compute + 2 * B.value * n_heads * seq_len.value * seq_len.value * _d_head
    swiglu_comp = 2 * B.value * seq_len.value * d_model.value * d_ff * 3  # d_ff is now a computed int
    lm_comp = 2 * B.value * seq_len.value * d_model.value * V.value
    o_proj_comp = 2 * B.value * seq_len.value * d_model.value * d_model.value

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
        "h": n_heads,  # n_heads is now a computed int
        "n_blocks": n_blocks.value,
        "d_ff": d_ff,  # d_ff is now a computed int
        "d_head": _d_head,
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
        "sp_size": fmt_size(sp_size),
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
def _(B, V, d_ff, d_head, d_model, ft_dtype, n_blocks, n_heads, seq_len, wt_dtype):
    """
    Comprehensive memory accounting for training (forward + backward).

    This traces every tensor that autograd saves during forward pass,
    plus gradient tensors and optimizer state.
    """
    # Bytes per element
    _dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "float8": 1}
    _wt_bytes = _dtype_bytes[wt_dtype.value]
    _ft_bytes = _dtype_bytes[ft_dtype.value]

    _B = B.value
    _S = seq_len.value
    _V = V.value
    _d = d_model.value
    _h = n_heads  # n_heads is now a computed int
    _L = n_blocks.value
    _dff = d_ff  # d_ff is now a computed int
    _dh = d_head.value  # d_head is now a slider input

    # =========================================================================
    # WEIGHTS (stored once, always in memory)
    # =========================================================================
    _weights = {
        "token_embeddings [V, d]": _V * _d * _wt_bytes,
        "lm_head [d, V]": _d * _V * _wt_bytes,
        "final_ln gamma [d]": _d * _wt_bytes,
    }

    # Per-block weights (× n_blocks)
    _per_block_weights = {
        "ln1 gamma [d]": _d * _wt_bytes,
        "ln2 gamma [d]": _d * _wt_bytes,
        "W_q [d, d]": _d * _d * _wt_bytes,
        "W_k [d, d]": _d * _d * _wt_bytes,
        "W_v [d, d]": _d * _d * _wt_bytes,
        "W_o [d, d]": _d * _d * _wt_bytes,
        "swiglu_w1 [d, dff]": _d * _dff * _wt_bytes,
        "swiglu_w2 [dff, d]": _dff * _d * _wt_bytes,
        "swiglu_w3 [d, dff]": _d * _dff * _wt_bytes,
    }

    _total_weights = sum(_weights.values()) + _L * sum(_per_block_weights.values())

    # =========================================================================
    # FORWARD ACTIVATIONS SAVED FOR BACKWARD (per block, accumulated)
    # These are kept alive by autograd until backward() completes
    # =========================================================================

    # TransformerLM.forward
    _fwd_global = {
        "input_indices [B, S]": _B * _S * 2,  # int16/int32
        "embeddings [B, S, d]": _B * _S * _d * _ft_bytes,
    }

    # Per-block forward activations (all L blocks accumulated!)
    # Following the pre-norm path in TransformerBlock.forward:
    _per_block_fwd = {
        # === Attention sublayer ===
        "input to block (residual) [B, S, d]": _B * _S * _d * _ft_bytes,
        "ln1 output [B, S, d]": _B * _S * _d * _ft_bytes,
        "ln1 rstd [B, S, 1]": _B * _S * _ft_bytes,  # saved for backward

        # QKV projection
        "QKV combined [B, S, 3d]": _B * _S * 3 * _d * _ft_bytes,

        # Reshaped heads
        "Q [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,
        "K [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,
        "V [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,

        # RoPE intermediates (sin/cos cached, but Q_rot, K_rot created)
        "Q after RoPE [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,
        "K after RoPE [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,

        # SDPA - THE BIG ONE (S² scaling)
        "QK^T / sqrt(dk) [B, h, S, S]": _B * _h * _S * _S * _ft_bytes,
        "masked scores [B, h, S, S]": _B * _h * _S * _S * _ft_bytes,
        # softmax intermediates (adjusted, exp, sum)
        "softmax adjusted [B, h, S, S]": _B * _h * _S * _S * _ft_bytes,
        "softmax exp [B, h, S, S]": _B * _h * _S * _S * _ft_bytes,
        "softmax output [B, h, S, S]": _B * _h * _S * _S * _ft_bytes,
        "attn @ V [B, h, S, dh]": _B * _h * _S * _dh * _ft_bytes,

        # Output projection
        "attn reshaped [B, S, d]": _B * _S * _d * _ft_bytes,
        "O proj output [B, S, d]": _B * _S * _d * _ft_bytes,
        "post-attn residual [B, S, d]": _B * _S * _d * _ft_bytes,

        # === FFN sublayer ===
        "ln2 output [B, S, d]": _B * _S * _d * _ft_bytes,
        "ln2 rstd [B, S, 1]": _B * _S * _ft_bytes,

        # SwiGLU: silu(w1(x)) * w3(x)
        "w1(x) [B, S, dff]": _B * _S * _dff * _ft_bytes,
        "w3(x) [B, S, dff]": _B * _S * _dff * _ft_bytes,
        "sigmoid(w1(x)) [B, S, dff]": _B * _S * _dff * _ft_bytes,  # silu backward
        "silu(w1(x)) [B, S, dff]": _B * _S * _dff * _ft_bytes,
        "silu * w3 [B, S, dff]": _B * _S * _dff * _ft_bytes,
        "w2 output [B, S, d]": _B * _S * _d * _ft_bytes,
    }

    # Final layers
    _fwd_final = {
        "final_ln output [B, S, d]": _B * _S * _d * _ft_bytes,
        "final_ln rstd [B, S, 1]": _B * _S * _ft_bytes,
        "lm_head logits [B, S, V]": _B * _S * _V * _ft_bytes,
    }

    _total_fwd_activations = (
        sum(_fwd_global.values()) + 
        _L * sum(_per_block_fwd.values()) + 
        sum(_fwd_final.values())
    )

    # =========================================================================
    # BACKWARD PASS - Gradient tensors
    # =========================================================================
    # Gradients have same size as weights
    _total_gradients = _total_weights

    # =========================================================================
    # OPTIMIZER STATE (AdamW)
    # Stores momentum (m) and variance (v) for each parameter
    # =========================================================================
    _optimizer_state = 2 * _total_weights  # m and v, same size as weights

    # =========================================================================
    # PEAK MEMORY ESTIMATE
    # Peak is at end of forward / start of backward:
    # weights + all fwd activations + starting to compute gradients
    # =========================================================================
    _peak_training = _total_weights + _total_fwd_activations + _total_gradients

    # Steady-state includes optimizer
    _steady_state = _total_weights + _optimizer_state + _total_gradients

    def _fmt(b):
        if b >= 1e9: return f"{b/1e9:.2f} GB"
        if b >= 1e6: return f"{b/1e6:.2f} MB"
        if b >= 1e3: return f"{b/1e3:.2f} KB"
        return f"{b} B"

    # Breakdown of where S² memory goes
    _s_squared_terms = _L * _B * _h * _S * _S * _ft_bytes * 5  # 5 S² tensors per block in softmax

    # =========================================================================
    # PARAMETER COUNT (for training efficiency)
    # =========================================================================
    _param_count_global = (
        _V * _d +  # token embeddings
        _d * _V +  # lm_head
        _d  # final ln
    )
    _param_count_per_block = (
        2 * _d +  # ln1, ln2
        4 * _d * _d +  # W_q, W_k, W_v, W_o
        3 * _d * _dff  # swiglu w1, w2, w3
    )
    _total_params = _param_count_global + _L * _param_count_per_block

    memory_accounting = {
        "total_weights": _fmt(_total_weights),
        "total_fwd_activations": _fmt(_total_fwd_activations),
        "total_gradients": _fmt(_total_gradients),
        "optimizer_state": _fmt(_optimizer_state),
        "peak_training": _fmt(_peak_training),
        "steady_state": _fmt(_steady_state),
        "s_squared_memory": _fmt(_s_squared_terms),
        "s_squared_raw": _s_squared_terms,
        "per_block_fwd": _fmt(_L * sum(_per_block_fwd.values())),
        "per_block_fwd_raw": sum(_per_block_fwd.values()),
        "attention_matrices_per_block": _fmt(_B * _h * _S * _S * _ft_bytes * 5),
        "swiglu_per_block": _fmt(_B * _S * _dff * _ft_bytes * 5),
        "n_blocks": _L,
        "per_block_weights_detail": _per_block_weights,
        "per_block_fwd_detail": _per_block_fwd,
        "total_params": _total_params,
        "batch_size": _B,
        "seq_len": _S,
    }
    return (memory_accounting,)


@app.cell
def _(B, memory_accounting, seq_len, train_steps):
    """Training efficiency calculations."""
    _B = B.value
    _S = seq_len.value
    _steps = train_steps.value
    _params = memory_accounting['total_params']

    # Tokens trained
    _tokens_per_step = _B * _S
    _total_tokens = _tokens_per_step * _steps

    # Compute estimate: ~6 * params * tokens for full training step
    # (2 for forward, 4 for backward with gradient computation)
    _total_flops = 6 * _params * _total_tokens

    # Chinchilla optimal: tokens ≈ 20 × params
    _chinchilla_tokens = 20 * _params
    _chinchilla_ratio = _total_tokens / _chinchilla_tokens

    def _fmt_num(n):
        if n >= 1e12: return f"{n/1e12:.2f}T"
        if n >= 1e9: return f"{n/1e9:.2f}B"
        if n >= 1e6: return f"{n/1e6:.2f}M"
        if n >= 1e3: return f"{n/1e3:.2f}K"
        return f"{n:.0f}"

    training_stats = {
        "total_params": _params,
        "total_params_fmt": _fmt_num(_params),
        "tokens_per_step": _tokens_per_step,
        "tokens_per_step_fmt": _fmt_num(_tokens_per_step),
        "total_tokens": _total_tokens,
        "total_tokens_fmt": _fmt_num(_total_tokens),
        "total_flops": _total_flops,
        "total_flops_fmt": _fmt_num(_total_flops),
        "train_steps": _steps,
        "chinchilla_tokens": _chinchilla_tokens,
        "chinchilla_tokens_fmt": _fmt_num(_chinchilla_tokens),
        "chinchilla_ratio": _chinchilla_ratio,
        "chinchilla_pct": f"{_chinchilla_ratio * 100:.1f}%",
    }
    return (training_stats,)


@app.cell
def _(B, budget_calc, memory_accounting, mo, seq_len, training_stats):
    def _fmt(b):
        if b >= 1e9: return f"{b/1e9:.2f} GB"
        if b >= 1e6: return f"{b/1e6:.2f} MB"
        if b >= 1e3: return f"{b/1e3:.2f} KB"
        return f"{b} B"

    _per_block_weights_table = "\n".join([
        f"| {k} | {_fmt(v)} |" for k, v in memory_accounting['per_block_weights_detail'].items()
    ])

    _per_block_fwd_table = "\n".join([
        f"| {k} | {_fmt(v)} |" for k, v in memory_accounting['per_block_fwd_detail'].items()
    ])

    # Check if current config fits budget
    _peak_bytes = float(memory_accounting['peak_training'].split()[0])
    _peak_unit = memory_accounting['peak_training'].split()[1]
    if _peak_unit == "GB":
        _peak_bytes *= 1e9
    elif _peak_unit == "MB":
        _peak_bytes *= 1e6

    _fits = _peak_bytes <= budget_calc['budget_bytes']
    _status = "✅ FITS" if _fits else "❌ EXCEEDS"
    _overage = "" if _fits else f" ({(_peak_bytes - budget_calc['budget_bytes'])/1e9:+.2f} GB)"

    # Chinchilla status
    _chinchilla_status = "✅" if training_stats['chinchilla_ratio'] >= 1.0 else "⚠️"

    _left_panel = mo.md(f"""
    ## Training Memory Accounting

    ### Peak Memory Estimates

    | Phase | Memory |
    |-------|--------|
    | **Peak (end of fwd / start of bwd)** | {memory_accounting['peak_training']} |
    | **Steady state** (weights + optim + grad) | {memory_accounting['steady_state']} |

    ### Summary

    | Category | Size |
    |----------|------|
    | **Weights** | {memory_accounting['total_weights']} |
    | **Forward Activations** (saved for backward) | {memory_accounting['total_fwd_activations']} |
    | **Gradients** (same as weights) | {memory_accounting['total_gradients']} |
    | **Optimizer State** (AdamW: 2× weights) | {memory_accounting['optimizer_state']} |

    ### S² Memory Breakdown

    - **Total S² memory** (all blocks): {memory_accounting['s_squared_memory']}
    - **Attention matrices per block**: {memory_accounting['attention_matrices_per_block']}
    - 5 tensors of [B, h, S, S] per block for softmax
    """)

    _middle_panel = mo.md(f"""
    ## Budget Calculator

    ### Status: {_status}{_overage}

    | | |
    |---|---|
    | **Budget** | {budget_calc['budget_gb']} GB |
    | **Peak** | {memory_accounting['peak_training']} |

    ### Fixed Costs

    | | |
    |---|---|
    | **Weights + Gradients** | {budget_calc['fixed_cost']} |
    | **Available for activations** | {budget_calc['headroom_for_activations']} |

    ### Maximum Values (within budget)

    | Parameter | Max | Current |
    |-----------|-----|---------|
    | **Batch (B)** | {budget_calc['max_B_at_current_S']} | {B.value} |
    | **Seq len (S)** | {budget_calc['max_S_at_B32']} | {seq_len.value} |

    *(Max B at current S; Max S at B=32)*
    """)

    _right_panel = mo.md(f"""
    ## Training Efficiency

    ### Model Size

    | | |
    |---|---|
    | **Parameters** | {training_stats['total_params_fmt']} |
    | **Steps** | {training_stats['train_steps']:,} |

    ### Tokens

    | | |
    |---|---|
    | **Per step** | {training_stats['tokens_per_step_fmt']} |
    | **Total** | {training_stats['total_tokens_fmt']} |

    ### Compute

    | | |
    |---|---|
    | **Est. FLOPs** | {training_stats['total_flops_fmt']} |

    ### Chinchilla Scaling {_chinchilla_status}

    | | |
    |---|---|
    | **Optimal tokens** | {training_stats['chinchilla_tokens_fmt']} |
    | **Your tokens** | {training_stats['total_tokens_fmt']} |
    | **Ratio** | {training_stats['chinchilla_pct']} |

    *Chinchilla: train on ~20× params tokens*
    """)

    mo.hstack([_left_panel, _middle_panel, _right_panel], justify="start", gap=3, widths=[1, 1, 1])
    return


@app.cell
def _(
    V,
    d_ff,
    d_head,
    d_model,
    ft_dtype,
    max_ram_gb,
    n_blocks,
    n_heads,
    seq_len,
    wt_dtype,
):
    """
    Calculate maximum B and S given a memory budget.

    Peak memory formula (simplified):
    peak = weights + fwd_activations + gradients

    Where fwd_activations has:
    - O(B*S*d) terms (linear in both B and S)
    - O(B*h*S²) terms (linear in B, quadratic in S) <- attention
    - O(B*S*d_ff) terms (linear in both)
    """
    import math

    _dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2, "float8": 1}
    _wt_bytes = _dtype_bytes[wt_dtype.value]
    _ft_bytes = _dtype_bytes[ft_dtype.value]

    _V = V.value
    _d = d_model.value
    _h = n_heads  # n_heads is now a computed int
    _L = n_blocks.value
    _dff = d_ff  # d_ff is now a computed int
    _dh = d_head.value  # d_head is now a slider input
    _S = seq_len.value

    _budget_bytes = max_ram_gb.value * 1e9

    # =========================================================================
    # Weight size (fixed, doesn't depend on B or S)
    # =========================================================================
    _weight_size = (
        2 * _V * _d * _wt_bytes +  # embeddings + lm_head
        _d * _wt_bytes +  # final ln
        _L * (2 * _d * _wt_bytes + 4 * _d * _d * _wt_bytes + 3 * _d * _dff * _wt_bytes)  # per block
    )
    _grad_size = _weight_size  # gradients same size as weights
    _fixed_cost = _weight_size + _grad_size

    # =========================================================================
    # Per-block activation coefficients (for solving max B and max S)
    # fwd_activations = global_terms + L * per_block_terms
    # =========================================================================

    # Coefficients for terms that scale as B*S*d (linear in B, linear in S)
    # Per block: input, ln1, ln1_rstd, QKV, Q, K, V, Q_rope, K_rope, attn_out, 
    #            attn_reshape, O_proj, residual, ln2, ln2_rstd, w2_out
    _coef_BSd = _L * (10 * _d + 3 * 3 * _d + 2) * _ft_bytes  # ~10 [B,S,d] tensors + QKV + rstd

    # Coefficients for terms that scale as B*h*S*dh = B*S*d (same as above, just reshaped)
    # Q, K, V, Q_rope, K_rope, attn@V = 6 tensors of [B,h,S,dh]
    _coef_BhSdh = _L * 6 * _ft_bytes  # per element

    # Coefficients for terms that scale as B*h*S² (linear in B, quadratic in S)
    # 5 attention matrices per block: QK^T, masked, softmax_adj, softmax_exp, softmax_out
    _coef_BhSS = _L * 5 * _h * _ft_bytes

    # Coefficients for terms that scale as B*S*d_ff (linear in B, linear in S)
    # 5 SwiGLU tensors: w1, w3, sigmoid, silu, silu*w3
    _coef_BSdff = _L * 5 * _dff * _ft_bytes

    # Global terms (outside blocks)
    # input_indices [B,S] (int16=2 bytes), embeddings [B,S,d], final_ln [B,S,d], logits [B,S,V]
    _coef_BS_global = 2 + _d * _ft_bytes + _d * _ft_bytes + _V * _ft_bytes

    # =========================================================================
    # Solve for max B (given current S)
    # peak = fixed + B * (S * coef_linear + S² * coef_quadratic)
    # B_max = (budget - fixed) / (S * coef_linear + S² * coef_quadratic)
    # =========================================================================
    _per_B_cost_at_S = (
        _S * (_coef_BSd + _coef_BhSdh * _dh + _coef_BSdff + _coef_BS_global) +
        _S * _S * _coef_BhSS
    )

    _available_for_activations = _budget_bytes - _fixed_cost
    _max_B = int(_available_for_activations / _per_B_cost_at_S) if _per_B_cost_at_S > 0 else 0
    _max_B = max(1, _max_B)

    # =========================================================================
    # Solve for max S (given current B from slider, which we need to get)
    # This is quadratic in S: a*S² + b*S + c = 0
    # where: a = B * coef_BhSS
    #        b = B * (coef_BSd + coef_BhSdh*dh + coef_BSdff + coef_BS_global)
    #        c = fixed - budget
    # =========================================================================
    # We need B value - let's use a reasonable default or compute for B=32
    _B_for_S_calc = 32  # typical batch size

    _a = _B_for_S_calc * _coef_BhSS
    _b = _B_for_S_calc * (_coef_BSd + _coef_BhSdh * _dh + _coef_BSdff + _coef_BS_global)
    _c = _fixed_cost - _budget_bytes

    # Quadratic formula: S = (-b + sqrt(b² - 4ac)) / 2a
    _discriminant = _b * _b - 4 * _a * _c
    if _discriminant >= 0 and _a > 0:
        _max_S = int((-_b + math.sqrt(_discriminant)) / (2 * _a))
        _max_S = max(128, _max_S)  # minimum reasonable sequence length
    else:
        _max_S = 128  # fallback

    # Round down to nearest 128 for practical use
    _max_S = (_max_S // 128) * 128

    # =========================================================================
    # Calculate what current config uses vs budget
    # =========================================================================
    def _fmt(b):
        if b >= 1e9: return f"{b/1e9:.2f} GB"
        if b >= 1e6: return f"{b/1e6:.2f} MB"
        return f"{b/1e3:.2f} KB"

    _headroom = _budget_bytes - _fixed_cost

    budget_calc = {
        "budget_gb": max_ram_gb.value,
        "budget_bytes": _budget_bytes,
        "fixed_cost": _fmt(_fixed_cost),
        "fixed_cost_bytes": _fixed_cost,
        "headroom_for_activations": _fmt(_headroom),
        "max_B_at_current_S": _max_B,
        "max_S_at_B32": _max_S,
        "current_S": _S,
        "per_B_cost_at_S": _fmt(_per_B_cost_at_S),
    }
    return (budget_calc,)


@app.cell(disabled=True)
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
    | attention [B,h,S,S] | {svg_vars['sp_size']}
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


@app.cell
def _(mo):
    svg_zoom = mo.ui.slider(25, 150, step=5, value=50, label="Zoom %", show_value=True)
    svg_zoom
    return (svg_zoom,)


@app.cell
def _(mo, svg_vars, svg_zoom):
    with open("notebooks/cs336_forward.svg") as f:
        svg = f.read()

    # Substitute all template variables from svg_vars
    for key, value in svg_vars.items():
        svg = svg.replace(f"{{{{{key}}}}}", str(value))

    # inject font styling into the SVG
    font_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    text {
      font-family: "Inter", system-ui, -apple-system, sans-serif;
    }
    </style>
    """
    svg = svg.replace(">", f">{font_style}", 1)

    # Wrap in a scrollable container with zoom
    zoom_pct = svg_zoom.value
    container_html = f"""
    <div style="
        width: 100%;
        height: 600px;
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        background: #fafafa;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    ">
        <div style="
            transform: scale({zoom_pct / 100});
            transform-origin: top left;
            padding: 16px;
        ">
            {svg}
        </div>
    </div>
    """

    mo.Html(container_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Targeted Benchmarks

    Now that we know what parts of the model are likely to cause bottlenecks, we can run some targeted benchmarks to assess where certain variables become problematic on our architecture

    ## Sequence length

    The attention mechanism involves a cacluation which scales quadratically with sequence length.  We expect that performance will quicly degrade after some critical sequence length.
    """)
    return


if __name__ == "__main__":
    app.run()
