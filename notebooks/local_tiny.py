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
    d_model = mo.ui.slider(128, 2048, step=128, value=512, label="d_model")
    n_layers = mo.ui.slider(2, 48, step=1, value=12, label="n_layers")
    lr = mo.ui.number(value=3e-4, step=1e-4, label="learning_rate")
    dropout = mo.ui.slider(0.0, 0.5, step=0.01, value=0.1, label="dropout")

    controls2 = mo.vstack([d_model, n_layers, lr, dropout])
    controls2


    return d_model, dropout, lr, n_layers


@app.cell
def _(d_model, dropout, lr, mo, n_layers):
    diagram2 = f"""
    flowchart LR
    A[Dataset] --> B[Tokenizer]
    B --> C[Model<br/>d_model={d_model.value}<br/>n_layers={n_layers.value}<br/>dropout={dropout.value}]
    C --> D[Train<br/>lr={lr.value}]
    D --> E[Eval]
    E --> F[Artifacts]
    """
    mo.mermaid(diagram2)
    return


@app.cell
def _(mo):
    a = mo.ui.slider(0, 100, value=30, label="A (0-100)")
    b = mo.ui.slider(0, 100, value=70, label="B (0-100)")
    c = mo.ui.number(value=1.5, label="C (float)")
    scale = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Scale")
    return a, b, c, scale


@app.cell
def _(a, b, c, mo, scale):
    weighted = (a.value * 0.6) + (b.value * 0.4)
    coupled = (a.value - b.value) * c.value
    blended = (weighted + coupled) * scale.value

    controls = mo.vstack([a, b, c, scale])
    outputs = mo.md(
        f"""
    **Inputs**
    - A = {a.value}
    - B = {b.value}
    - C = {c.value}
    - Scale = {scale.value}

    **Outputs**
    - Weighted(A,B) = 0.6*A + 0.4*B = {weighted:.2f}
    - Coupled(A,B,C) = (A - B) * C = {coupled:.2f}
    - Blended = (Weighted + Coupled) * Scale = {blended:.2f}
    """
    )

    controls, weighted, coupled, blended
    return


@app.cell
def _(mo):
    diagram1 = '''

    flowchart TB
        subgraph Input
            tokens["Input Tokens<br/>[batch, seq]"]
        end

        subgraph Embeddings
            emb["Token Embedding<br/>weight: [vocab_size, d_model]"]
        end

        tokens --> emb
        emb --> |"[batch, seq, d_model]"| block1

        subgraph block1["Transformer Block ×N"]
            direction TB
        
            subgraph attn_branch["Multi-Head Self-Attention"]
                ln1["RMSNorm<br/>weight: [d_model]"]
            
                subgraph projections["QKV Projections"]
                    qproj["W_Q: [d_model, d_model]"]
                    kproj["W_K: [d_model, d_model]"]
                    vproj["W_V: [d_model, d_model]"]
                end
            
                split["Split into heads<br/>[batch, num_heads, seq, d_head]<br/>d_head = d_model / num_heads"]
            
                rope["RoPE<br/>cos/sin: [max_seq_len, d_head/2]"]
            
                sdpa["Scaled Dot-Product Attention<br/>QK^T/√d_k → softmax → ×V<br/>+ Causal Mask"]
            
                concat["Concat Heads<br/>[batch, seq, d_model]"]
            
                oproj["W_O: [d_model, d_model]"]
            end
        
            res1(("+"))
        
            subgraph ffn_branch["SwiGLU FFN"]
                ln2["RMSNorm<br/>weight: [d_model]"]
                w1["W1: [d_ff, d_model]"]
                w3["W3: [d_ff, d_model]"]
                silu_act["SiLU(W1·x)"]
                gate["⊙ (element-wise)"]
                w2["W2: [d_model, d_ff]"]
            end
        
            res2(("+"))
        
            ln1 --> projections
            projections --> split
            split --> rope
            rope --> sdpa
            sdpa --> concat
            concat --> oproj
            oproj --> res1
        
            res1 --> ln2
            ln2 --> w1
            ln2 --> w3
            w1 --> silu_act
            silu_act --> gate
            w3 --> gate
            gate --> w2
            w2 --> res2
        end

        subgraph Output
            final_ln["Final RMSNorm<br/>weight: [d_model]"]
            lm_head["LM Head (Linear)<br/>weight: [vocab_size, d_model]"]
            logits["Output Logits<br/>[batch, seq, vocab_size]"]
        end

        block1 --> |"[batch, seq, d_model]"| final_ln
        final_ln --> lm_head
        lm_head --> logits

        %% Residual connections
        emb -.->|residual| res1
        res1 -.->|residual| res2
    '''

    mo.mermaid(diagram1)

    return


@app.cell
def _(mo):
    with open("notebooks/cs336_forward.svg") as f:
        svg = f.read()


    svg = svg.replace("{{B}}", str("5"))

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
