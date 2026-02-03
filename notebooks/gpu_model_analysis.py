import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import yaml
    import math
    from pathlib import Path
    return Path, go, math, mo, pl, px, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GPU Model Analysis: 4090 Training Comparison

    This notebook analyzes Models A and B trained on an RTX 4090 with bfloat16 mixed precision.
    It compares theoretical architecture calculations with observed wandb results, and projects
    scaling limits for the 4090.

    ## RTX 4090 Specifications

    | Spec | Value |
    |------|-------|
    | **VRAM** | 24 GB GDDR6X |
    | **Memory Bandwidth** | 1008 GB/s |
    | **FP32 TFLOPS** | 82.6 |
    | **TF32 TFLOPS** | 165 (tensor cores) |
    | **FP16/BF16 TFLOPS** | 330 (tensor cores) |
    | **Architecture** | Ada Lovelace (AD102) |
    | **CUDA Cores** | 16,384 |
    | **Tensor Cores** | 512 (4th gen) |
    """)
    return


@app.cell
def _():
    # RTX 4090 hardware specs
    GPU_SPECS = {
        "name": "RTX 4090",
        "vram_gb": 24,
        "memory_bandwidth_gb_s": 1008,
        "fp32_tflops": 82.6,
        "tf32_tflops": 165,
        "bf16_tflops": 330,
        "fp16_tflops": 330,
    }
    return (GPU_SPECS,)


@app.cell
def _(Path, yaml):
    # Load model configurations from YAML
    config_path = Path("configs/models.yaml")
    with open(config_path) as f:
        model_configs = yaml.safe_load(f)

    model_a_config = model_configs["model_a"]
    model_b_config = model_configs["model_b"]
    return config_path, model_a_config, model_b_config, model_configs


@app.cell
def _(mo, wandb_df):
    # Extract the actual configs used during training from wandb
    # This handles cases where YAML was updated after training
    
    _rows = list(wandb_df.iter_rows(named=True))
    
    trained_configs = {}
    for _row in _rows:
        _name = _row["run_name"]
        _ms = _row["config.model_settings"]
        trained_configs[_name] = {
            "name": _name,
            "batch_size": _row["config.batch_size"],
            "model_settings": {
                "vocab_size": _ms["vocab_size"],
                "d_model": _ms["d_model"],
                "num_heads": _ms["num_heads"],
                "num_layers": _ms["num_layers"],
                "d_ff": _ms["d_ff"],
                "context_length": _ms["context_length"],
            }
        }
    
    # Find Model A and Model B by name pattern
    model_a_trained = None
    model_b_trained = None
    for _name, _cfg in trained_configs.items():
        if "Model_A" in _name:
            model_a_trained = _cfg
        elif "Model_B" in _name:
            model_b_trained = _cfg
    
    mo.md(f"""
### Configurations Extracted from WandB

Found {len(trained_configs)} training runs. Using configs from wandb data for analysis.
""")
    return model_a_trained, model_b_trained, trained_configs


@app.cell
def _(Path, pl):
    # Load wandb results
    parquet_path = Path("notebooks/benchmark_results/gpu_initial.parquet")
    wandb_df = pl.read_parquet(parquet_path)
    wandb_df
    return parquet_path, wandb_df


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Configuration Summary
    """)
    return


@app.cell
def _(mo, model_a_config, model_a_trained, model_b_config, model_b_trained):
    # Use trained configs if available, otherwise fall back to YAML
    _model_a = model_a_trained if model_a_trained else model_a_config
    _model_b = model_b_trained if model_b_trained else model_b_config
    
    def _fmt_config(cfg, source="yaml"):
        ms = cfg["model_settings"]
        return f"""
| Parameter | Value |
|-----------|-------|
| **Name** | {cfg['name']} |
| **Source** | {source} |
| **batch_size** | {cfg['batch_size']} |
| **vocab_size** | {ms['vocab_size']:,} |
| **d_model** | {ms['d_model']} |
| **num_heads** | {ms['num_heads']} |
| **d_head** | {ms['d_model'] // ms['num_heads']} |
| **num_layers** | {ms['num_layers']} |
| **d_ff** | {ms['d_ff']} |
| **context_length** | {ms['context_length']} |
| **FFN ratio** | {ms['d_ff'] / ms['d_model']:.2f}× |
"""

    _a_source = "wandb" if model_a_trained else "yaml"
    _b_source = "wandb" if model_b_trained else "yaml"

    mo.hstack([
        mo.vstack([
            mo.md("### Model A (Wide & Shallow)"),
            mo.md(_fmt_config(_model_a, _a_source)),
        ]),
        mo.vstack([
            mo.md("### Model B (Narrow & Deep)"),
            mo.md(_fmt_config(_model_b, _b_source)),
        ]),
    ], justify="start", gap=4)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Theoretical Architecture Analysis

    ### Parameter Count

    For a transformer LM:
    - **Embeddings**: $2 \times V \times d_{model}$ (token embeddings + LM head, tied or not)
    - **Per-layer**: $4 \times d_{model}^2$ (Q, K, V, O projections) + $3 \times d_{model} \times d_{ff}$ (SwiGLU) + $2 \times d_{model}$ (RMSNorm)
    - **Final**: $d_{model}$ (final RMSNorm)

    ### FLOPs per Forward Pass

    For each token in a batch:
    - **QKV projection**: $2 \times 3 \times d_{model}^2$ (3 matrices)
    - **Attention**: $2 \times S \times d_{model}$ (QK^T) + $2 \times S \times d_{model}$ (softmax @ V)
    - **Output projection**: $2 \times d_{model}^2$
    - **SwiGLU FFN**: $2 \times 3 \times d_{model} \times d_{ff}$
    - **LM head**: $2 \times d_{model} \times V$

    Training is ~3× forward (forward + backward ≈ 3×, or 6× params × tokens rule).
    """)
    return


@app.cell
def _():
    def calculate_model_params(vocab_size, d_model, num_heads, num_layers, d_ff):
        """Calculate total parameter count for transformer LM."""
        # Embeddings (token + LM head - may be tied but count both)
        emb_params = 2 * vocab_size * d_model
        
        # Final RMSNorm
        final_ln_params = d_model
        
        # Per layer
        per_layer_params = (
            2 * d_model +                    # 2 RMSNorms (ln1, ln2)
            4 * d_model * d_model +          # Q, K, V, O projections
            3 * d_model * d_ff               # SwiGLU (w1, w2, w3)
        )
        
        total_params = emb_params + final_ln_params + num_layers * per_layer_params
        return {
            "total": total_params,
            "embeddings": emb_params,
            "per_layer": per_layer_params,
            "final_ln": final_ln_params,
            "total_M": total_params / 1e6,
        }


    def calculate_forward_flops(
        batch_size, seq_len, vocab_size, d_model, num_heads, num_layers, d_ff
    ):
        """
        Calculate FLOPs for one forward pass through the model.
        
        Uses 2× for matmuls (multiply-accumulate counts as 2 ops).
        """
        B, S, V, d, h, L, dff = batch_size, seq_len, vocab_size, d_model, num_heads, num_layers, d_ff
        d_head = d // h
        
        # Embedding lookup is essentially free (indexing)
        
        # Per-layer FLOPs
        per_layer_flops = (
            # RMSNorm 1: 2 * B * S * d (compute variance + normalize)
            2 * B * S * d +
            
            # QKV projection: 2 * B * S * d * 3d = 6 * B * S * d^2
            2 * B * S * d * (3 * d) +
            
            # RoPE: ~2 * B * h * S * d_head (sin/cos multiply)
            2 * B * h * S * d_head +
            
            # Attention QK^T: 2 * B * h * S * S * d_head
            2 * B * h * S * S * d_head +
            
            # Softmax: ~3 * B * h * S * S (exp, sum, div)
            3 * B * h * S * S +
            
            # Attention @ V: 2 * B * h * S * S * d_head
            2 * B * h * S * S * d_head +
            
            # Output projection: 2 * B * S * d * d
            2 * B * S * d * d +
            
            # RMSNorm 2: 2 * B * S * d
            2 * B * S * d +
            
            # SwiGLU: w1, w3 projections + silu + element-wise mult + w2
            # w1: 2 * B * S * d * dff
            # w3: 2 * B * S * d * dff
            # silu: ~2 * B * S * dff (sigmoid + mult)
            # mult: B * S * dff
            # w2: 2 * B * S * dff * d
            2 * B * S * d * dff +  # w1
            2 * B * S * d * dff +  # w3
            3 * B * S * dff +      # silu + mult
            2 * B * S * dff * d    # w2
        )
        
        # Final RMSNorm: 2 * B * S * d
        final_norm_flops = 2 * B * S * d
        
        # LM head: 2 * B * S * d * V
        lm_head_flops = 2 * B * S * d * V
        
        total_forward_flops = L * per_layer_flops + final_norm_flops + lm_head_flops
        
        return {
            "total": total_forward_flops,
            "per_layer": per_layer_flops,
            "lm_head": lm_head_flops,
            "attention_per_layer": 2 * B * h * S * S * d_head * 2 + 3 * B * h * S * S,  # QK^T + softmax + attn@V
            "ffn_per_layer": 2 * B * S * d * dff * 3 + 3 * B * S * dff,  # w1, w3, w2 + silu
            "total_TFLOPs": total_forward_flops / 1e12,
        }


    def calculate_training_step_flops(forward_flops):
        """Training step ≈ 3× forward (forward + backward with gradient computation)."""
        return {
            "forward": forward_flops,
            "backward": forward_flops * 2,  # ~2× forward for backward
            "total": forward_flops * 3,
            "total_TFLOPs": forward_flops * 3 / 1e12,
        }
    return (
        calculate_forward_flops,
        calculate_model_params,
        calculate_training_step_flops,
    )


@app.cell
def _(
    calculate_forward_flops,
    calculate_model_params,
    calculate_training_step_flops,
    model_a_config,
    model_a_trained,
    model_b_config,
    model_b_trained,
):
    # Calculate theoretical values for both models
    # Use trained configs if available (from wandb), otherwise fall back to YAML
    def _analyze_model(config):
        ms = config["model_settings"]
        batch_size = config["batch_size"]
        
        params = calculate_model_params(
            vocab_size=ms["vocab_size"],
            d_model=ms["d_model"],
            num_heads=ms["num_heads"],
            num_layers=ms["num_layers"],
            d_ff=ms["d_ff"],
        )
        
        forward_flops = calculate_forward_flops(
            batch_size=batch_size,
            seq_len=ms["context_length"],
            vocab_size=ms["vocab_size"],
            d_model=ms["d_model"],
            num_heads=ms["num_heads"],
            num_layers=ms["num_layers"],
            d_ff=ms["d_ff"],
        )
        
        training_flops = calculate_training_step_flops(forward_flops["total"])
        
        # Tokens per step
        tokens_per_step = batch_size * ms["context_length"]
        
        return {
            "name": config["name"],
            "params": params,
            "forward_flops": forward_flops,
            "training_flops": training_flops,
            "tokens_per_step": tokens_per_step,
            "config": config,
        }

    _model_a = model_a_trained if model_a_trained else model_a_config
    _model_b = model_b_trained if model_b_trained else model_b_config
    
    model_a_analysis = _analyze_model(_model_a)
    model_b_analysis = _analyze_model(_model_b)
    return model_a_analysis, model_b_analysis


@app.cell
def _(mo, model_a_analysis, model_b_analysis):
    def _fmt(n):
        if n >= 1e12: return f"{n/1e12:.2f}T"
        if n >= 1e9: return f"{n/1e9:.2f}G"
        if n >= 1e6: return f"{n/1e6:.2f}M"
        if n >= 1e3: return f"{n/1e3:.2f}K"
        return f"{n:.0f}"

    a = model_a_analysis
    b = model_b_analysis
    
    mo.md(f"""
    ### Theoretical Calculations Summary

    | Metric | Model A | Model B | Ratio (A/B) |
    |--------|---------|---------|-------------|
    | **Parameters** | {_fmt(a['params']['total'])} | {_fmt(b['params']['total'])} | {a['params']['total']/b['params']['total']:.2f}× |
    | **Embedding params** | {_fmt(a['params']['embeddings'])} | {_fmt(b['params']['embeddings'])} | {a['params']['embeddings']/b['params']['embeddings']:.2f}× |
    | **Per-layer params** | {_fmt(a['params']['per_layer'])} | {_fmt(b['params']['per_layer'])} | {a['params']['per_layer']/b['params']['per_layer']:.2f}× |
    | **Tokens/step** | {_fmt(a['tokens_per_step'])} | {_fmt(b['tokens_per_step'])} | {a['tokens_per_step']/b['tokens_per_step']:.2f}× |
    | **Forward FLOPs** | {_fmt(a['forward_flops']['total'])} | {_fmt(b['forward_flops']['total'])} | {a['forward_flops']['total']/b['forward_flops']['total']:.2f}× |
    | **Training step FLOPs** | {_fmt(a['training_flops']['total'])} | {_fmt(b['training_flops']['total'])} | {a['training_flops']['total']/b['training_flops']['total']:.2f}× |
    | **Attention FLOPs/layer** | {_fmt(a['forward_flops']['attention_per_layer'])} | {_fmt(b['forward_flops']['attention_per_layer'])} | {a['forward_flops']['attention_per_layer']/b['forward_flops']['attention_per_layer']:.2f}× |
    | **FFN FLOPs/layer** | {_fmt(a['forward_flops']['ffn_per_layer'])} | {_fmt(b['forward_flops']['ffn_per_layer'])} | {a['forward_flops']['ffn_per_layer']/b['forward_flops']['ffn_per_layer']:.2f}× |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Memory Analysis

    ### Weight Memory (bfloat16 = 2 bytes per param)

    For mixed-precision training:
    - **Master weights**: FP32 (4 bytes) - stored by optimizer
    - **Model weights**: BF16 (2 bytes) - used for forward/backward
    - **Gradients**: BF16 (2 bytes) - same as model weights
    - **Optimizer state**: 2× FP32 for AdamW (m and v)

    Total memory ≈ weights × (4 + 2 + 2 + 8) = 16 bytes per parameter

    ### Activation Memory

    Key tensors during training (per layer):
    - Attention matrices: $B \times h \times S^2$ (× 5 for softmax intermediates)
    - Hidden states: $B \times S \times d_{model}$ (multiple copies for residuals, layernorm)
    - FFN intermediates: $B \times S \times d_{ff}$
    """)
    return


@app.cell
def _():
    def calculate_memory_breakdown(
        batch_size, seq_len, vocab_size, d_model, num_heads, num_layers, d_ff,
        precision="bf16"
    ):
        """
        Calculate memory breakdown for training.
        
        Returns memory in bytes.
        """
        B, S, V, d, h, L, dff = batch_size, seq_len, vocab_size, d_model, num_heads, num_layers, d_ff
        d_head = d // h
        
        # Bytes per element
        bytes_map = {"fp32": 4, "bf16": 2, "fp16": 2}
        act_bytes = bytes_map.get(precision, 2)  # activations in training precision
        
        # For mixed precision: optimizer keeps FP32 master weights
        # Model weights + gradients in BF16, optimizer state in FP32
        
        # === Parameter memory ===
        total_params = (
            2 * V * d +                    # embeddings + LM head
            d +                            # final RMSNorm
            L * (2 * d + 4 * d * d + 3 * d * dff)  # per-layer
        )
        
        # Mixed precision memory per parameter:
        # - BF16 model weights: 2 bytes
        # - BF16 gradients: 2 bytes
        # - FP32 master weights: 4 bytes (in optimizer)
        # - FP32 optimizer m: 4 bytes
        # - FP32 optimizer v: 4 bytes
        param_memory = total_params * (2 + 2 + 4 + 4 + 4)  # 16 bytes/param
        
        # === Activation memory (saved for backward) ===
        
        # Global activations
        global_act = (
            B * S * 2 +                    # input indices (int16)
            B * S * d * act_bytes +        # embeddings
            B * S * d * act_bytes +        # final layernorm output
            B * S * V * act_bytes          # logits (largest tensor!)
        )
        
        # Per-layer activations (accumulated across all layers)
        per_layer_act = (
            # Residual connections and layer outputs
            3 * B * S * d * act_bytes +    # input, post-attn, post-ffn
            
            # LayerNorm intermediates
            2 * B * S * act_bytes +        # rstd for both norms
            
            # QKV and projections
            B * S * 3 * d * act_bytes +    # QKV combined
            
            # Attention heads (reshaped)
            4 * B * h * S * d_head * act_bytes +  # Q, K, V, attn output
            
            # Attention matrices (S² scaling!)
            5 * B * h * S * S * act_bytes +  # QK^T, masked, softmax intermediates
            
            # FFN (SwiGLU)
            5 * B * S * dff * act_bytes     # w1, w3, silu, gate*up, w2 input
        )
        
        total_activation = global_act + L * per_layer_act
        
        # Peak memory estimate (end of forward, start of backward)
        peak_memory = param_memory + total_activation
        
        return {
            "total_params": total_params,
            "param_memory_bytes": param_memory,
            "param_memory_gb": param_memory / 1e9,
            "activation_memory_bytes": total_activation,
            "activation_memory_gb": total_activation / 1e9,
            "peak_memory_bytes": peak_memory,
            "peak_memory_gb": peak_memory / 1e9,
            "attention_memory_per_layer_bytes": 5 * B * h * S * S * act_bytes,
            "attention_memory_per_layer_gb": 5 * B * h * S * S * act_bytes / 1e9,
            "logits_memory_bytes": B * S * V * act_bytes,
            "logits_memory_gb": B * S * V * act_bytes / 1e9,
        }
    return (calculate_memory_breakdown,)


@app.cell
def _(
    calculate_memory_breakdown,
    model_a_config,
    model_a_trained,
    model_b_config,
    model_b_trained,
):
    def _mem_analysis(config):
        ms = config["model_settings"]
        return calculate_memory_breakdown(
            batch_size=config["batch_size"],
            seq_len=ms["context_length"],
            vocab_size=ms["vocab_size"],
            d_model=ms["d_model"],
            num_heads=ms["num_heads"],
            num_layers=ms["num_layers"],
            d_ff=ms["d_ff"],
            precision="bf16",
        )

    _model_a = model_a_trained if model_a_trained else model_a_config
    _model_b = model_b_trained if model_b_trained else model_b_config
    
    model_a_memory = _mem_analysis(_model_a)
    model_b_memory = _mem_analysis(_model_b)
    return model_a_memory, model_b_memory


@app.cell
def _(mo, model_a_memory, model_b_memory, model_a_trained, model_b_trained, model_a_config, model_b_config):
    a_mem = model_a_memory
    b_mem = model_b_memory
    
    # Get layer counts from trained configs or fallback
    _model_a = model_a_trained if model_a_trained else model_a_config
    _model_b = model_b_trained if model_b_trained else model_b_config
    _a_layers = _model_a["model_settings"]["num_layers"]
    _b_layers = _model_b["model_settings"]["num_layers"]
    
    # Calculate per-layer breakdown
    _a_per_layer_act = a_mem['activation_memory_gb'] / _a_layers if _a_layers > 0 else 0
    _b_per_layer_act = b_mem['activation_memory_gb'] / _b_layers if _b_layers > 0 else 0
    _a_per_layer_param = (a_mem['param_memory_gb'] - 0.15) / _a_layers  # subtract embedding overhead estimate
    _b_per_layer_param = (b_mem['param_memory_gb'] - 0.08) / _b_layers
    
    mo.md(f"""
    ### Memory Estimates (BF16 Mixed Precision)

    | Component | Model A | Model B |
    |-----------|---------|---------|
    | **Parameters** | {a_mem['total_params']/1e6:.1f}M | {b_mem['total_params']/1e6:.1f}M |
    | **Parameter memory** | {a_mem['param_memory_gb']:.2f} GB | {b_mem['param_memory_gb']:.2f} GB |
    | **Activation memory** | {a_mem['activation_memory_gb']:.2f} GB | {b_mem['activation_memory_gb']:.2f} GB |
    | **Peak memory (est.)** | {a_mem['peak_memory_gb']:.2f} GB | {b_mem['peak_memory_gb']:.2f} GB |
    | **Attention mem/layer** | {a_mem['attention_memory_per_layer_gb']*1000:.1f} MB | {b_mem['attention_memory_per_layer_gb']*1000:.1f} MB |
    | **Logits tensor** | {a_mem['logits_memory_gb']*1000:.1f} MB | {b_mem['logits_memory_gb']*1000:.1f} MB |
    
    ### Per-Layer Resource Breakdown

    | Component | Model A ({_a_layers} layers) | Model B ({_b_layers} layers) |
    |-----------|------------------------------|------------------------------|
    | **Activation mem/layer** | {_a_per_layer_act*1000:.1f} MB | {_b_per_layer_act*1000:.1f} MB |
    | **Param mem/layer** | {_a_per_layer_param*1000:.1f} MB | {_b_per_layer_param*1000:.1f} MB |
    | **Total layers** | {_a_layers} | {_b_layers} |
    | **Layer depth ratio** | 1.0× | {_b_layers/_a_layers:.1f}× |
    | **Attention matrices/layer** | B×{_model_a['model_settings']['num_heads']}×S² | B×{_model_b['model_settings']['num_heads']}×S² |
    | **FFN intermediate/layer** | B×S×{_model_a['model_settings']['d_ff']} | B×S×{_model_b['model_settings']['d_ff']} |
    
    *Note: Peak memory is theoretical; actual may differ due to PyTorch allocator behavior, 
    gradient checkpointing, and CUDA memory fragmentation. Per-layer activation memory includes
    attention matrices (O(S²)), hidden states, and FFN intermediates.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## WandB Observed Results

    Compare theoretical calculations with actual training metrics from the 4090 runs.
    """)
    return


@app.cell
def _(pl, wandb_df):
    # Extract model settings from the struct column
    wandb_expanded = wandb_df.with_columns([
        pl.col("config.model_settings").struct.field("d_model").alias("d_model"),
        pl.col("config.model_settings").struct.field("d_ff").alias("d_ff"),
        pl.col("config.model_settings").struct.field("num_heads").alias("num_heads"),
        pl.col("config.model_settings").struct.field("num_layers").alias("num_layers"),
        pl.col("config.model_settings").struct.field("vocab_size").alias("vocab_size"),
        pl.col("config.model_settings").struct.field("context_length").alias("context_length"),
    ])

    # Select relevant columns for analysis
    wandb_summary = wandb_expanded.select([
        "run_name",
        "config.batch_size",
        "config.precision",
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "vocab_size",
        "context_length",
        "Throughput/Tokens per sec",
        "Memory/Max allocated (GB)",
        "Time/Total step",
        "Time/Forward",
        "Time/Backward",
        "Eval/Best loss",
        "Loss",
    ]).rename({
        "config.batch_size": "batch_size",
        "config.precision": "precision",
        "Throughput/Tokens per sec": "tokens_per_sec",
        "Memory/Max allocated (GB)": "memory_gb",
        "Time/Total step": "step_time_s",
        "Time/Forward": "forward_time_s",
        "Time/Backward": "backward_time_s",
        "Eval/Best loss": "best_val_loss",
        "Loss": "final_train_loss",
    })
    
    wandb_summary
    return wandb_expanded, wandb_summary


@app.cell
def _(mo, pl, wandb_summary):
    # Create comparison table
    _rows = list(wandb_summary.iter_rows(named=True))
    
    _output = None
    if len(_rows) >= 2:
        _a_obs = _rows[0]  # Model A
        _b_obs = _rows[1]  # Model B
        
        _output = mo.md(f"""
### Observed Training Metrics

| Metric | Model A | Model B | Ratio (A/B) |
|--------|---------|---------|-------------|
| **Throughput** | {_a_obs['tokens_per_sec']:,.0f} tok/s | {_b_obs['tokens_per_sec']:,.0f} tok/s | {_a_obs['tokens_per_sec']/_b_obs['tokens_per_sec']:.2f}× |
| **Memory (max)** | {_a_obs['memory_gb']:.2f} GB | {_b_obs['memory_gb']:.2f} GB | {_a_obs['memory_gb']/_b_obs['memory_gb']:.2f}× |
| **Step time** | {_a_obs['step_time_s']*1000:.1f} ms | {_b_obs['step_time_s']*1000:.1f} ms | {_a_obs['step_time_s']/_b_obs['step_time_s']:.2f}× |
| **Forward time** | {_a_obs['forward_time_s']*1000:.1f} ms | {_b_obs['forward_time_s']*1000:.1f} ms | {_a_obs['forward_time_s']/_b_obs['forward_time_s']:.2f}× |
| **Backward time** | {_a_obs['backward_time_s']*1000:.1f} ms | {_b_obs['backward_time_s']*1000:.1f} ms | {_a_obs['backward_time_s']/_b_obs['backward_time_s']:.2f}× |
| **Best val loss** | {_a_obs['best_val_loss']:.4f} | {_b_obs['best_val_loss']:.4f} | - |
| **Final train loss** | {_a_obs['final_train_loss']:.4f} | {_b_obs['final_train_loss']:.4f} | - |
""")
    else:
        _output = mo.md("*Not enough runs in the data to compare.*")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Theoretical vs Observed Comparison
    """)
    return


@app.cell
def _(
    GPU_SPECS,
    mo,
    model_a_analysis,
    model_a_memory,
    model_b_analysis,
    model_b_memory,
    wandb_summary,
):
    _rows = list(wandb_summary.iter_rows(named=True))
    _output = None
    
    if len(_rows) >= 2:
        _a_obs = _rows[0]
        _b_obs = _rows[1]
        
        # Calculate theoretical throughput based on compute
        # throughput = tokens_per_step / step_time
        # step_time = training_flops / (gpu_tflops * efficiency)
        
        # Theoretical max throughput (100% compute utilization)
        _a_theoretical_step_time = model_a_analysis['training_flops']['total'] / (GPU_SPECS['bf16_tflops'] * 1e12)
        _b_theoretical_step_time = model_b_analysis['training_flops']['total'] / (GPU_SPECS['bf16_tflops'] * 1e12)
        
        _a_theoretical_throughput = model_a_analysis['tokens_per_step'] / _a_theoretical_step_time
        _b_theoretical_throughput = model_b_analysis['tokens_per_step'] / _b_theoretical_step_time
        
        # Compute utilization (MFU - Model FLOPS Utilization)
        _a_mfu = (model_a_analysis['training_flops']['total'] / _a_obs['step_time_s']) / (GPU_SPECS['bf16_tflops'] * 1e12)
        _b_mfu = (model_b_analysis['training_flops']['total'] / _b_obs['step_time_s']) / (GPU_SPECS['bf16_tflops'] * 1e12)
        
        _output = mo.md(f"""
### Throughput Analysis

| Metric | Model A | Model B |
|--------|---------|---------|
| **Theoretical max throughput** | {_a_theoretical_throughput:,.0f} tok/s | {_b_theoretical_throughput:,.0f} tok/s |
| **Observed throughput** | {_a_obs['tokens_per_sec']:,.0f} tok/s | {_b_obs['tokens_per_sec']:,.0f} tok/s |
| **Efficiency** | {_a_obs['tokens_per_sec']/_a_theoretical_throughput*100:.1f}% | {_b_obs['tokens_per_sec']/_b_theoretical_throughput*100:.1f}% |
| **MFU (Model FLOPS Utilization)** | {_a_mfu*100:.1f}% | {_b_mfu*100:.1f}% |

### Memory Analysis

| Metric | Model A | Model B |
|--------|---------|---------|
| **Theoretical peak memory** | {model_a_memory['peak_memory_gb']:.2f} GB | {model_b_memory['peak_memory_gb']:.2f} GB |
| **Observed max memory** | {_a_obs['memory_gb']:.2f} GB | {_b_obs['memory_gb']:.2f} GB |
| **Ratio (Obs/Theo)** | {_a_obs['memory_gb']/model_a_memory['peak_memory_gb']:.2f}× | {_b_obs['memory_gb']/model_b_memory['peak_memory_gb']:.2f}× |
| **Headroom to 24GB** | {24 - _a_obs['memory_gb']:.2f} GB | {24 - _b_obs['memory_gb']:.2f} GB |

*Note: MFU < 50% is typical for small models due to memory bandwidth limitations and kernel launch overhead.*
""")
    else:
        _output = mo.md("*Not enough wandb data to analyze.*")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4090 Scaling Limits Analysis

    Let's project when the RTX 4090 will hit memory or compute bottlenecks as we scale model size.
    """)
    return


@app.cell
def _(GPU_SPECS, calculate_memory_breakdown, mo, pl):
    # Scaling analysis: vary d_model, num_layers, batch_size, seq_len
    # and find the limits

    scaling_configs = []

    # Base configuration to vary from
    base = {
        "vocab_size": 10000,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 1365,  # ~8/3 ratio
        "batch_size": 32,
        "seq_len": 256,
    }

    # Vary d_model (width)
    for d in [256, 384, 512, 768, 1024, 1536, 2048]:
        _cfg = base.copy()
        _cfg["d_model"] = d
        _cfg["num_heads"] = d // 64  # keep d_head=64
        _cfg["d_ff"] = int(d * 8 / 3 / 64) * 64  # SwiGLU ratio, rounded
        _cfg["vary"] = "d_model"
        scaling_configs.append(_cfg)

    # Vary num_layers (depth)
    for L in [2, 4, 6, 8, 12, 16, 24, 32]:
        _cfg = base.copy()
        _cfg["num_layers"] = L
        _cfg["vary"] = "num_layers"
        scaling_configs.append(_cfg)

    # Vary batch_size
    for B in [8, 16, 32, 64, 128, 256, 512]:
        _cfg = base.copy()
        _cfg["batch_size"] = B
        _cfg["vary"] = "batch_size"
        scaling_configs.append(_cfg)

    # Vary seq_len (context length)
    for S in [128, 256, 512, 1024, 2048, 4096]:
        _cfg = base.copy()
        _cfg["seq_len"] = S
        _cfg["vary"] = "seq_len"
        scaling_configs.append(_cfg)

    # Calculate memory for each config
    scaling_results = []
    for _cfg in scaling_configs:
        _mem = calculate_memory_breakdown(
            batch_size=_cfg["batch_size"],
            seq_len=_cfg["seq_len"],
            vocab_size=_cfg["vocab_size"],
            d_model=_cfg["d_model"],
            num_heads=_cfg["num_heads"],
            num_layers=_cfg["num_layers"],
            d_ff=_cfg["d_ff"],
            precision="bf16",
        )
        scaling_results.append({
            **_cfg,
            "peak_memory_gb": _mem["peak_memory_gb"],
            "param_memory_gb": _mem["param_memory_gb"],
            "activation_memory_gb": _mem["activation_memory_gb"],
            "total_params_M": _mem["total_params"] / 1e6,
            "fits_4090": _mem["peak_memory_gb"] <= GPU_SPECS["vram_gb"],
        })

    scaling_df = pl.DataFrame(scaling_results)
    mo.md("### Scaling Analysis Configurations Generated")
    return base, scaling_configs, scaling_df, scaling_results


@app.cell
def _(GPU_SPECS, go, mo, px, scaling_df):
    # Plot memory vs parameter variation
    _df_dmodel = scaling_df.filter(scaling_df["vary"] == "d_model")
    
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_df_dmodel["d_model"].to_list(),
        y=_df_dmodel["peak_memory_gb"].to_list(),
        mode="lines+markers",
        name="Peak Memory",
        line=dict(color="blue"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_dmodel["d_model"].to_list(),
        y=_df_dmodel["param_memory_gb"].to_list(),
        mode="lines+markers",
        name="Param Memory",
        line=dict(color="green", dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_dmodel["d_model"].to_list(),
        y=_df_dmodel["activation_memory_gb"].to_list(),
        mode="lines+markers",
        name="Activation Memory",
        line=dict(color="orange", dash="dot"),
    ))
    _fig.add_hline(y=GPU_SPECS["vram_gb"], line_dash="dash", line_color="red",
                   annotation_text="4090 VRAM (24GB)")
    
    _fig.update_layout(
        title="Memory Scaling with d_model (batch=32, seq=256, layers=6)",
        xaxis_title="d_model",
        yaxis_title="Memory (GB)",
        height=400,
    )
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(GPU_SPECS, go, mo, scaling_df):
    # Plot memory vs num_layers
    _df_layers = scaling_df.filter(scaling_df["vary"] == "num_layers")
    
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_df_layers["num_layers"].to_list(),
        y=_df_layers["peak_memory_gb"].to_list(),
        mode="lines+markers",
        name="Peak Memory",
        line=dict(color="blue"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_layers["num_layers"].to_list(),
        y=_df_layers["param_memory_gb"].to_list(),
        mode="lines+markers",
        name="Param Memory",
        line=dict(color="green", dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_layers["num_layers"].to_list(),
        y=_df_layers["activation_memory_gb"].to_list(),
        mode="lines+markers",
        name="Activation Memory",
        line=dict(color="orange", dash="dot"),
    ))
    _fig.add_hline(y=GPU_SPECS["vram_gb"], line_dash="dash", line_color="red",
                   annotation_text="4090 VRAM (24GB)")
    
    _fig.update_layout(
        title="Memory Scaling with Depth (batch=32, seq=256, d_model=512)",
        xaxis_title="num_layers",
        yaxis_title="Memory (GB)",
        height=400,
    )
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(GPU_SPECS, go, mo, scaling_df):
    # Plot memory vs batch_size
    _df_batch = scaling_df.filter(scaling_df["vary"] == "batch_size")
    
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_df_batch["batch_size"].to_list(),
        y=_df_batch["peak_memory_gb"].to_list(),
        mode="lines+markers",
        name="Peak Memory",
        line=dict(color="blue"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_batch["batch_size"].to_list(),
        y=_df_batch["activation_memory_gb"].to_list(),
        mode="lines+markers",
        name="Activation Memory",
        line=dict(color="orange", dash="dot"),
    ))
    _fig.add_hline(y=GPU_SPECS["vram_gb"], line_dash="dash", line_color="red",
                   annotation_text="4090 VRAM (24GB)")
    
    _fig.update_layout(
        title="Memory Scaling with Batch Size (seq=256, d_model=512, layers=6)",
        xaxis_title="batch_size",
        yaxis_title="Memory (GB)",
        height=400,
    )
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(GPU_SPECS, go, mo, scaling_df):
    # Plot memory vs seq_len (most important for attention!)
    _df_seq = scaling_df.filter(scaling_df["vary"] == "seq_len")
    
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_df_seq["seq_len"].to_list(),
        y=_df_seq["peak_memory_gb"].to_list(),
        mode="lines+markers",
        name="Peak Memory",
        line=dict(color="blue"),
    ))
    _fig.add_trace(go.Scatter(
        x=_df_seq["seq_len"].to_list(),
        y=_df_seq["activation_memory_gb"].to_list(),
        mode="lines+markers",
        name="Activation Memory",
        line=dict(color="orange", dash="dot"),
    ))
    _fig.add_hline(y=GPU_SPECS["vram_gb"], line_dash="dash", line_color="red",
                   annotation_text="4090 VRAM (24GB)")
    
    _fig.update_layout(
        title="Memory Scaling with Sequence Length (batch=32, d_model=512, layers=6)",
        xaxis_title="seq_len",
        yaxis_title="Memory (GB)",
        height=400,
        xaxis_type="log",
    )
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Interactive Scaling Calculator

    Explore different configurations to find the limits on the 4090.
    """)
    return


@app.cell
def _(mo):
    # Interactive sliders for scaling exploration
    calc_batch = mo.ui.slider(1, 256, step=1, value=32, label="Batch Size", show_value=True)
    calc_seq = mo.ui.slider(64, 8192, step=64, value=256, label="Sequence Length", show_value=True)
    calc_d_model = mo.ui.slider(128, 4096, step=64, value=512, label="d_model", show_value=True)
    calc_layers = mo.ui.slider(1, 48, step=1, value=6, label="num_layers", show_value=True)
    calc_d_ff_ratio = mo.ui.slider(1.0, 8.0, step=0.5, value=2.67, label="d_ff ratio", show_value=True)
    calc_vocab = mo.ui.slider(1000, 100000, step=1000, value=10000, label="Vocab Size", show_value=True)

    mo.vstack([
        mo.md("### Configuration"),
        mo.hstack([calc_batch, calc_seq, calc_vocab], justify="start", gap=2),
        mo.hstack([calc_d_model, calc_layers, calc_d_ff_ratio], justify="start", gap=2),
    ])
    return (
        calc_batch,
        calc_d_ff_ratio,
        calc_d_model,
        calc_layers,
        calc_seq,
        calc_vocab,
    )


@app.cell
def _(
    GPU_SPECS,
    calc_batch,
    calc_d_ff_ratio,
    calc_d_model,
    calc_layers,
    calc_seq,
    calc_vocab,
    calculate_forward_flops,
    calculate_memory_breakdown,
    calculate_model_params,
    calculate_training_step_flops,
    math,
    mo,
):
    # Calculate for current config
    _d = calc_d_model.value
    _d_ff = int(_d * calc_d_ff_ratio.value / 64) * 64  # Round to multiple of 64
    _h = max(1, _d // 64)  # d_head = 64
    
    _params = calculate_model_params(
        vocab_size=calc_vocab.value,
        d_model=_d,
        num_heads=_h,
        num_layers=calc_layers.value,
        d_ff=_d_ff,
    )
    
    _mem = calculate_memory_breakdown(
        batch_size=calc_batch.value,
        seq_len=calc_seq.value,
        vocab_size=calc_vocab.value,
        d_model=_d,
        num_heads=_h,
        num_layers=calc_layers.value,
        d_ff=_d_ff,
        precision="bf16",
    )
    
    _forward = calculate_forward_flops(
        batch_size=calc_batch.value,
        seq_len=calc_seq.value,
        vocab_size=calc_vocab.value,
        d_model=_d,
        num_heads=_h,
        num_layers=calc_layers.value,
        d_ff=_d_ff,
    )
    
    _training = calculate_training_step_flops(_forward["total"])
    
    _tokens_per_step = calc_batch.value * calc_seq.value
    
    # Theoretical step time (100% MFU)
    _theo_step_time = _training["total"] / (GPU_SPECS["bf16_tflops"] * 1e12)
    
    # Realistic step time (assume 35% MFU for small models)
    _real_mfu = 0.35
    _real_step_time = _theo_step_time / _real_mfu
    _real_throughput = _tokens_per_step / _real_step_time
    
    # Memory status
    _mem_pct = _mem["peak_memory_gb"] / GPU_SPECS["vram_gb"] * 100
    _mem_status = "✅" if _mem_pct <= 90 else ("⚠️" if _mem_pct <= 100 else "❌")
    
    def _fmt(n):
        if n >= 1e12: return f"{n/1e12:.2f}T"
        if n >= 1e9: return f"{n/1e9:.2f}G"
        if n >= 1e6: return f"{n/1e6:.2f}M"
        if n >= 1e3: return f"{n/1e3:.2f}K"
        return f"{n:.0f}"

    mo.md(f"""
    ### Results for Current Configuration

    | Metric | Value |
    |--------|-------|
    | **d_ff (computed)** | {_d_ff} |
    | **num_heads (d_head=64)** | {_h} |
    | **Total Parameters** | {_fmt(_params['total'])} ({_params['total_M']:.1f}M) |
    | **Tokens per step** | {_tokens_per_step:,} |

    #### Memory {_mem_status}

    | Component | Value | % of 24GB |
    |-----------|-------|-----------|
    | **Peak memory** | {_mem['peak_memory_gb']:.2f} GB | {_mem_pct:.1f}% |
    | **Parameter memory** | {_mem['param_memory_gb']:.2f} GB | {_mem['param_memory_gb']/24*100:.1f}% |
    | **Activation memory** | {_mem['activation_memory_gb']:.2f} GB | {_mem['activation_memory_gb']/24*100:.1f}% |
    | **Headroom** | {24 - _mem['peak_memory_gb']:.2f} GB | - |

    #### Compute

    | Metric | Value |
    |--------|-------|
    | **Forward FLOPs** | {_fmt(_forward['total'])} |
    | **Training step FLOPs** | {_fmt(_training['total'])} |
    | **Theoretical step time** | {_theo_step_time*1000:.2f} ms (100% MFU) |
    | **Realistic step time** | {_real_step_time*1000:.1f} ms ({_real_mfu*100:.0f}% MFU) |
    | **Est. throughput** | {_real_throughput:,.0f} tok/s |
    
    *Estimates assume {_real_mfu*100:.0f}% MFU, typical for small-medium models.*
    """)
    return


@app.cell
def _(GPU_SPECS, calculate_memory_breakdown, mo):
    # Find maximum configurations that fit in 4090
    
    def _find_max_param(vary_param, base_config, max_val, step):
        """Binary search for max value that fits in VRAM."""
        low, high = step, max_val
        best = low
        
        while low <= high:
            mid = ((low + high) // 2 // step) * step  # Round to step
            cfg = base_config.copy()
            cfg[vary_param] = mid
            
            # Adjust dependent params for d_model
            if vary_param == "d_model":
                cfg["num_heads"] = max(1, mid // 64)
                cfg["d_ff"] = int(mid * 2.67 / 64) * 64
            
            mem = calculate_memory_breakdown(**cfg, precision="bf16")
            
            if mem["peak_memory_gb"] <= GPU_SPECS["vram_gb"] * 0.95:  # 95% margin
                best = mid
                low = mid + step
            else:
                high = mid - step
        
        return best

    _base = {
        "batch_size": 32,
        "seq_len": 256,
        "vocab_size": 10000,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 1365,
    }

    # Find limits
    _max_batch = _find_max_param("batch_size", _base, 1024, 8)
    _max_seq = _find_max_param("seq_len", _base, 16384, 128)
    _max_d_model = _find_max_param("d_model", _base, 8192, 64)
    _max_layers = _find_max_param("num_layers", _base, 96, 1)

    mo.md(f"""
    ### 4090 Maximum Configurations (with 5% safety margin)

    Starting from base config: batch=32, seq=256, d_model=512, layers=6

    | Parameter | Max Value | Limiting Factor |
    |-----------|-----------|-----------------|
    | **batch_size** | {_max_batch} | Activation memory (linear) |
    | **seq_len** | {_max_seq} | Attention matrices (O(S²)) |
    | **d_model** | {_max_d_model} | Param + activation memory |
    | **num_layers** | {_max_layers} | Activation memory (linear) |

    *Note: These are one-at-a-time limits. Increasing multiple parameters simultaneously 
    will hit the memory wall sooner.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## MPS (Apple Silicon) Analysis
    
    This section analyzes the same Model A and Model B architectures trained on Apple Silicon 
    using MPS (Metal Performance Shaders) with float32 precision.
    
    ### Apple Silicon MPS Specifications
    
    MPS provides GPU acceleration on Apple Silicon but with different characteristics than CUDA:
    
    | Aspect | MPS (Apple Silicon) | CUDA (4090) |
    |--------|---------------------|-------------|
    | **Memory** | Unified (shared with CPU) | Dedicated VRAM |
    | **Precision** | FP32 (BF16 limited) | BF16/FP16/TF32/FP32 |
    | **Tensor Cores** | None | 512 (4th gen) |
    | **Memory Bandwidth** | ~400 GB/s (M3 Max) | 1008 GB/s |
    | **Architecture** | Apple GPU | Ada Lovelace |
    """)
    return


@app.cell
def _(Path, pl):
    # Load MPS benchmark results
    mps_parquet_path = Path("notebooks/benchmark_results/mps_initial.parquet")
    mps_df = pl.read_parquet(mps_parquet_path)
    mps_df
    return mps_df, mps_parquet_path


@app.cell
def _(mps_df, mo):
    # Extract MPS configs
    _rows = list(mps_df.iter_rows(named=True))
    
    mps_trained_configs = {}
    for _row in _rows:
        _name = _row["run_name"]
        _ms = _row["config.model_settings"]
        mps_trained_configs[_name] = {
            "name": _name,
            "batch_size": _row["config.batch_size"],
            "device": _row["config.device"],
            "model_settings": {
                "vocab_size": _ms["vocab_size"],
                "d_model": _ms["d_model"],
                "num_heads": _ms["num_heads"],
                "num_layers": _ms["num_layers"],
                "d_ff": _ms["d_ff"],
                "context_length": _ms["context_length"],
            }
        }
    
    # Find Model A and Model B
    mps_model_a = None
    mps_model_b = None
    for _name, _cfg in mps_trained_configs.items():
        if "Model_A" in _name:
            mps_model_a = _cfg
        elif "Model_B" in _name:
            mps_model_b = _cfg
    
    mo.md(f"""
### MPS Training Configurations

Found {len(mps_trained_configs)} MPS training runs with FP32 precision.
""")
    return mps_model_a, mps_model_b, mps_trained_configs


@app.cell
def _(pl, mps_df):
    # Extract MPS metrics
    mps_expanded = mps_df.with_columns([
        pl.col("config.model_settings").struct.field("d_model").alias("d_model"),
        pl.col("config.model_settings").struct.field("d_ff").alias("d_ff"),
        pl.col("config.model_settings").struct.field("num_heads").alias("num_heads"),
        pl.col("config.model_settings").struct.field("num_layers").alias("num_layers"),
        pl.col("config.model_settings").struct.field("vocab_size").alias("vocab_size"),
        pl.col("config.model_settings").struct.field("context_length").alias("context_length"),
    ])

    mps_summary = mps_expanded.select([
        "run_name",
        "config.batch_size",
        "config.device",
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "vocab_size",
        "context_length",
        "Throughput/Tokens per sec",
        "Memory/Current allocated (GB)",
        "Time/Total step",
        "Time/Forward",
        "Time/Backward",
        "Eval/Best loss",
        "Loss",
    ]).rename({
        "config.batch_size": "batch_size",
        "config.device": "device",
        "Throughput/Tokens per sec": "tokens_per_sec",
        "Memory/Current allocated (GB)": "memory_gb",
        "Time/Total step": "step_time_s",
        "Time/Forward": "forward_time_s",
        "Time/Backward": "backward_time_s",
        "Eval/Best loss": "best_val_loss",
        "Loss": "final_train_loss",
    })
    
    mps_summary
    return mps_expanded, mps_summary


@app.cell
def _(mo, mps_summary):
    # MPS observed metrics table
    _rows = list(mps_summary.iter_rows(named=True))
    
    _output = None
    if len(_rows) >= 2:
        _a_obs = _rows[0]  # Model A
        _b_obs = _rows[1]  # Model B
        
        _output = mo.md(f"""
### MPS Observed Training Metrics (FP32)

| Metric | Model A | Model B | Ratio (A/B) |
|--------|---------|---------|-------------|
| **Throughput** | {_a_obs['tokens_per_sec']:,.0f} tok/s | {_b_obs['tokens_per_sec']:,.0f} tok/s | {_a_obs['tokens_per_sec']/_b_obs['tokens_per_sec']:.2f}× |
| **Memory (current)** | {_a_obs['memory_gb']:.2f} GB | {_b_obs['memory_gb']:.2f} GB | {_a_obs['memory_gb']/_b_obs['memory_gb']:.2f}× |
| **Step time** | {_a_obs['step_time_s']*1000:.1f} ms | {_b_obs['step_time_s']*1000:.1f} ms | {_a_obs['step_time_s']/_b_obs['step_time_s']:.2f}× |
| **Forward time** | {_a_obs['forward_time_s']*1000:.1f} ms | {_b_obs['forward_time_s']*1000:.1f} ms | {_a_obs['forward_time_s']/_b_obs['forward_time_s']:.2f}× |
| **Backward time** | {_a_obs['backward_time_s']*1000:.1f} ms | {_b_obs['backward_time_s']*1000:.1f} ms | {_a_obs['backward_time_s']/_b_obs['backward_time_s']:.2f}× |
| **Best val loss** | {_a_obs['best_val_loss']:.4f} | {_b_obs['best_val_loss']:.4f} | - |
| **Final train loss** | {_a_obs['final_train_loss']:.4f} | {_b_obs['final_train_loss']:.4f} | - |

*Note: MPS uses FP32 precision (4 bytes/param) vs BF16 (2 bytes/param) on CUDA, 
resulting in higher memory usage per parameter but potentially better numerical precision.*
""")
    else:
        _output = mo.md("*Not enough MPS runs in the data to compare.*")
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## MPS vs RTX 4090 Comparison
    
    Direct comparison of training performance between Apple Silicon MPS (FP32) and 
    NVIDIA RTX 4090 (BF16 mixed precision).
    """)
    return


@app.cell
def _(mo, mps_summary, wandb_summary):
    # Cross-platform comparison
    _mps_rows = list(mps_summary.iter_rows(named=True))
    _gpu_rows = list(wandb_summary.iter_rows(named=True))
    
    _output = None
    if len(_mps_rows) >= 2 and len(_gpu_rows) >= 2:
        _mps_a = _mps_rows[0]  # MPS Model A
        _mps_b = _mps_rows[1]  # MPS Model B
        _gpu_a = _gpu_rows[0]  # GPU Model A
        _gpu_b = _gpu_rows[1]  # GPU Model B
        
        _output = mo.md(f"""
### Cross-Platform Performance Comparison

#### Model A (Wide & Shallow: d_model=768, 2 layers)

| Metric | MPS (FP32) | 4090 (BF16) | Speedup (4090/MPS) |
|--------|------------|-------------|---------------------|
| **Throughput** | {_mps_a['tokens_per_sec']:,.0f} tok/s | {_gpu_a['tokens_per_sec']:,.0f} tok/s | **{_gpu_a['tokens_per_sec']/_mps_a['tokens_per_sec']:.1f}×** |
| **Step time** | {_mps_a['step_time_s']*1000:.1f} ms | {_gpu_a['step_time_s']*1000:.2f} ms | {_mps_a['step_time_s']/_gpu_a['step_time_s']:.1f}× |
| **Forward time** | {_mps_a['forward_time_s']*1000:.1f} ms | {_gpu_a['forward_time_s']*1000:.2f} ms | {_mps_a['forward_time_s']/_gpu_a['forward_time_s']:.1f}× |
| **Backward time** | {_mps_a['backward_time_s']*1000:.1f} ms | {_gpu_a['backward_time_s']*1000:.2f} ms | {_mps_a['backward_time_s']/_gpu_a['backward_time_s']:.1f}× |
| **Memory** | {_mps_a['memory_gb']:.2f} GB | {_gpu_a['memory_gb']:.2f} GB | {_mps_a['memory_gb']/_gpu_a['memory_gb']:.2f}× |
| **Best val loss** | {_mps_a['best_val_loss']:.4f} | {_gpu_a['best_val_loss']:.4f} | - |

#### Model B (Narrow & Deep: d_model=384, 12 layers)

| Metric | MPS (FP32) | 4090 (BF16) | Speedup (4090/MPS) |
|--------|------------|-------------|---------------------|
| **Throughput** | {_mps_b['tokens_per_sec']:,.0f} tok/s | {_gpu_b['tokens_per_sec']:,.0f} tok/s | **{_gpu_b['tokens_per_sec']/_mps_b['tokens_per_sec']:.1f}×** |
| **Step time** | {_mps_b['step_time_s']*1000:.1f} ms | {_gpu_b['step_time_s']*1000:.1f} ms | {_mps_b['step_time_s']/_gpu_b['step_time_s']:.1f}× |
| **Forward time** | {_mps_b['forward_time_s']*1000:.1f} ms | {_gpu_b['forward_time_s']*1000:.1f} ms | {_mps_b['forward_time_s']/_gpu_b['forward_time_s']:.1f}× |
| **Backward time** | {_mps_b['backward_time_s']*1000:.1f} ms | {_gpu_b['backward_time_s']*1000:.1f} ms | {_mps_b['backward_time_s']/_gpu_b['backward_time_s']:.1f}× |
| **Memory** | {_mps_b['memory_gb']:.2f} GB | {_gpu_b['memory_gb']:.2f} GB | {_mps_b['memory_gb']/_gpu_b['memory_gb']:.2f}× |
| **Best val loss** | {_mps_b['best_val_loss']:.4f} | {_gpu_b['best_val_loss']:.4f} | - |

### Key Observations

1. **Throughput Gap**: The 4090 achieves **{_gpu_a['tokens_per_sec']/_mps_a['tokens_per_sec']:.0f}× higher throughput** on Model A 
   and **{_gpu_b['tokens_per_sec']/_mps_b['tokens_per_sec']:.0f}× higher** on Model B.

2. **Depth Penalty on MPS**: Model B (12 layers) shows a larger performance gap vs the 4090 compared to 
   Model A (2 layers). The 4090's tensor cores and higher memory bandwidth handle deep sequential 
   computation more efficiently.

3. **Memory Efficiency**: Despite using FP32 (2× bytes per param), MPS shows lower memory usage 
   due to unified memory architecture and different allocation patterns. The 4090's reported memory 
   includes CUDA allocator overhead.

4. **Backward Pass Scaling**: The backward pass shows the largest speedup differential, likely due to 
   the 4090's tensor cores being particularly efficient for gradient computation with BF16.

5. **Model Quality**: Both platforms achieve similar validation losses, confirming that BF16 mixed 
   precision on the 4090 doesn't significantly impact model quality for these architectures.
""")
    else:
        _output = mo.md("*Not enough data for cross-platform comparison.*")
    _output
    return


@app.cell
def _(go, mo, mps_summary, wandb_summary):
    # Visualization: MPS vs 4090 throughput comparison
    _mps_rows = list(mps_summary.iter_rows(named=True))
    _gpu_rows = list(wandb_summary.iter_rows(named=True))
    
    _output = None
    if len(_mps_rows) >= 2 and len(_gpu_rows) >= 2:
        _models = ["Model A\n(Wide)", "Model B\n(Deep)"]
        _mps_throughput = [_mps_rows[0]['tokens_per_sec'], _mps_rows[1]['tokens_per_sec']]
        _gpu_throughput = [_gpu_rows[0]['tokens_per_sec'], _gpu_rows[1]['tokens_per_sec']]
        
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="MPS (FP32)",
            x=_models,
            y=_mps_throughput,
            text=[f"{t:,.0f}" for t in _mps_throughput],
            textposition="outside",
            marker_color="orange",
        ))
        _fig.add_trace(go.Bar(
            name="4090 (BF16)",
            x=_models,
            y=_gpu_throughput,
            text=[f"{t:,.0f}" for t in _gpu_throughput],
            textposition="outside",
            marker_color="blue",
        ))
        
        _fig.update_layout(
            title="Training Throughput: MPS vs RTX 4090",
            xaxis_title="Model Architecture",
            yaxis_title="Tokens per Second",
            barmode="group",
            height=450,
            yaxis_type="log",
        )
        
        _output = mo.ui.plotly(_fig)
    _output
    return


@app.cell
def _(go, mo, mps_summary, wandb_summary):
    # Visualization: Step time breakdown
    _mps_rows = list(mps_summary.iter_rows(named=True))
    _gpu_rows = list(wandb_summary.iter_rows(named=True))
    
    _output = None
    if len(_mps_rows) >= 2 and len(_gpu_rows) >= 2:
        _categories = ["MPS Model A", "4090 Model A", "MPS Model B", "4090 Model B"]
        _forward = [
            _mps_rows[0]['forward_time_s']*1000,
            _gpu_rows[0]['forward_time_s']*1000,
            _mps_rows[1]['forward_time_s']*1000,
            _gpu_rows[1]['forward_time_s']*1000,
        ]
        _backward = [
            _mps_rows[0]['backward_time_s']*1000,
            _gpu_rows[0]['backward_time_s']*1000,
            _mps_rows[1]['backward_time_s']*1000,
            _gpu_rows[1]['backward_time_s']*1000,
        ]
        _other = [
            (_mps_rows[0]['step_time_s'] - _mps_rows[0]['forward_time_s'] - _mps_rows[0]['backward_time_s'])*1000,
            (_gpu_rows[0]['step_time_s'] - _gpu_rows[0]['forward_time_s'] - _gpu_rows[0]['backward_time_s'])*1000,
            (_mps_rows[1]['step_time_s'] - _mps_rows[1]['forward_time_s'] - _mps_rows[1]['backward_time_s'])*1000,
            (_gpu_rows[1]['step_time_s'] - _gpu_rows[1]['forward_time_s'] - _gpu_rows[1]['backward_time_s'])*1000,
        ]
        
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Forward", x=_categories, y=_forward, marker_color="green"))
        _fig.add_trace(go.Bar(name="Backward", x=_categories, y=_backward, marker_color="red"))
        _fig.add_trace(go.Bar(name="Other (optimizer, etc.)", x=_categories, y=_other, marker_color="gray"))
        
        _fig.update_layout(
            title="Step Time Breakdown: MPS vs RTX 4090",
            xaxis_title="Platform / Model",
            yaxis_title="Time (ms)",
            barmode="stack",
            height=450,
        )
        
        _output = mo.ui.plotly(_fig)
    _output
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Scaling for nsys/ncu Profiling Exercises
    
    This section discusses how to scale the model configurations to create interesting profiling 
    scenarios on the RTX 4090 - configurations that saturate and exceed system limits to reveal 
    bottlenecks visible in NVIDIA Nsight Systems (nsys) and Nsight Compute (ncu).
    """)
    return


@app.cell
def _(GPU_SPECS, calculate_memory_breakdown, mo, pl):
    # Define profiling-oriented configurations
    profiling_configs = []
    
    # Base models for reference
    base_a = {"name": "Model A (baseline)", "batch_size": 32, "seq_len": 256, 
              "d_model": 768, "num_heads": 12, "num_layers": 2, "d_ff": 2048, "vocab_size": 10000}
    base_b = {"name": "Model B (baseline)", "batch_size": 32, "seq_len": 256,
              "d_model": 384, "num_heads": 12, "num_layers": 12, "d_ff": 1024, "vocab_size": 10000}
    
    profiling_configs.append(base_a)
    profiling_configs.append(base_b)
    
    # Scenario 1: Memory bandwidth bound (large batch, small model)
    profiling_configs.append({
        "name": "Bandwidth Bound",
        "description": "Large batch with small model - memory bandwidth limited",
        "batch_size": 256, "seq_len": 256, "d_model": 384, "num_heads": 6, 
        "num_layers": 4, "d_ff": 1024, "vocab_size": 10000,
        "profile_interest": "Memory bandwidth saturation, low compute utilization"
    })
    
    # Scenario 2: Compute bound (moderate batch, larger model)
    profiling_configs.append({
        "name": "Compute Bound",
        "description": "Larger model with moderate batch - compute limited",
        "batch_size": 32, "seq_len": 256, "d_model": 1536, "num_heads": 24,
        "num_layers": 8, "d_ff": 4096, "vocab_size": 10000,
        "profile_interest": "High tensor core utilization, matmul dominance"
    })
    
    # Scenario 3: Attention memory explosion (long sequence)
    profiling_configs.append({
        "name": "Attention Memory Stress",
        "description": "Long sequence length - O(S²) attention memory",
        "batch_size": 16, "seq_len": 2048, "d_model": 512, "num_heads": 8,
        "num_layers": 6, "d_ff": 1365, "vocab_size": 10000,
        "profile_interest": "Attention kernel time, memory allocation patterns"
    })
    
    # Scenario 4: Near OOM (push to memory limit)
    profiling_configs.append({
        "name": "Near OOM",
        "description": "Configuration near 24GB VRAM limit",
        "batch_size": 64, "seq_len": 512, "d_model": 1024, "num_heads": 16,
        "num_layers": 12, "d_ff": 2730, "vocab_size": 10000,
        "profile_interest": "Memory fragmentation, allocator behavior, potential OOM"
    })
    
    # Scenario 5: Deep sequential (many layers)
    profiling_configs.append({
        "name": "Deep Sequential",
        "description": "Many layers - sequential kernel launches",
        "batch_size": 32, "seq_len": 256, "d_model": 512, "num_heads": 8,
        "num_layers": 32, "d_ff": 1365, "vocab_size": 10000,
        "profile_interest": "Kernel launch overhead, layer-to-layer dependencies"
    })
    
    # Scenario 6: Wide FFN (SwiGLU stress)
    profiling_configs.append({
        "name": "Wide FFN",
        "description": "Large FFN ratio - SwiGLU memory and compute",
        "batch_size": 32, "seq_len": 256, "d_model": 768, "num_heads": 12,
        "num_layers": 6, "d_ff": 6144, "vocab_size": 10000,
        "profile_interest": "FFN kernel performance, intermediate tensor sizes"
    })
    
    # Scenario 7: Exceed VRAM (intentional OOM)
    profiling_configs.append({
        "name": "OOM Trigger",
        "description": "Intentionally exceeds 24GB - for OOM debugging",
        "batch_size": 128, "seq_len": 1024, "d_model": 1024, "num_heads": 16,
        "num_layers": 16, "d_ff": 2730, "vocab_size": 10000,
        "profile_interest": "OOM error handling, memory allocation failure points"
    })
    
    # Calculate memory for each
    profiling_results = []
    for _cfg in profiling_configs:
        _mem = calculate_memory_breakdown(
            batch_size=_cfg["batch_size"],
            seq_len=_cfg["seq_len"],
            vocab_size=_cfg["vocab_size"],
            d_model=_cfg["d_model"],
            num_heads=_cfg["num_heads"],
            num_layers=_cfg["num_layers"],
            d_ff=_cfg["d_ff"],
            precision="bf16",
        )
        profiling_results.append({
            "name": _cfg["name"],
            "batch_size": _cfg["batch_size"],
            "seq_len": _cfg["seq_len"],
            "d_model": _cfg["d_model"],
            "num_layers": _cfg["num_layers"],
            "d_ff": _cfg["d_ff"],
            "params_M": _mem["total_params"] / 1e6,
            "peak_memory_gb": _mem["peak_memory_gb"],
            "fits_4090": _mem["peak_memory_gb"] <= GPU_SPECS["vram_gb"],
            "vram_pct": _mem["peak_memory_gb"] / GPU_SPECS["vram_gb"] * 100,
            "description": _cfg.get("description", ""),
            "profile_interest": _cfg.get("profile_interest", ""),
        })
    
    profiling_df = pl.DataFrame(profiling_results)
    profiling_df
    return profiling_configs, profiling_df, profiling_results


@app.cell
def _(mo, profiling_results):
    # Format profiling scenarios table
    _rows = []
    for r in profiling_results:
        _status = "✅" if r["fits_4090"] else "❌"
        _rows.append(f"| {r['name']} | {r['batch_size']} | {r['seq_len']} | {r['d_model']} | {r['num_layers']} | {r['params_M']:.1f}M | {r['peak_memory_gb']:.1f} GB | {r['vram_pct']:.0f}% | {_status} |")
    
    _table = "\n".join(_rows)
    
    mo.md(f"""
### Profiling Scenario Configurations

| Name | Batch | Seq | d_model | Layers | Params | Est. Memory | VRAM % | Fits |
|------|-------|-----|---------|--------|--------|-------------|--------|------|
{_table}

### Scenario Descriptions and Profiling Interest

""" + "\n".join([
    f"**{r['name']}**: {r.get('description', 'N/A')}\n- *Profile interest*: {r.get('profile_interest', 'N/A')}\n"
    for r in profiling_results if r.get('description')
]))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### nsys/ncu Profiling Recommendations
    
    #### Using Nsight Systems (nsys) for Timeline Analysis
    
    ```bash
    # Basic profiling command
    nsys profile -o profile_output python train.py --config <config>
    
    # With CUDA and cuDNN tracing
    nsys profile --trace=cuda,cudnn,nvtx -o detailed_profile python train.py
    
    # Capture specific duration (first 10 seconds)
    nsys profile --duration=10 -o short_profile python train.py
    ```
    
    **What to look for in nsys:**
    - Kernel launch gaps (CPU-GPU synchronization overhead)
    - Memory transfer patterns (H2D, D2H copies)
    - Kernel concurrency (are operations overlapping?)
    - Stream utilization (multiple CUDA streams?)
    
    #### Using Nsight Compute (ncu) for Kernel Analysis
    
    ```bash
    # Profile specific kernels
    ncu --set full -o kernel_analysis python train.py
    
    # Target specific kernel patterns
    ncu --kernel-name "volta_fp16_s*" -o attention_kernels python train.py
    
    # Memory throughput analysis
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
        -o throughput_analysis python train.py
    ```
    
    **What to look for in ncu:**
    - Achieved occupancy vs theoretical
    - Memory throughput (% of peak)
    - Compute throughput (tensor core utilization)
    - Warp stalls (memory, execution, synchronization)
    
    #### Recommended Profiling Exercises
    
    1. **Memory Bandwidth Exercise**: Compare "Bandwidth Bound" vs "Compute Bound" configs
       - Observe memory throughput saturation in ncu
       - Compare kernel execution time vs memory transfer time in nsys
    
    2. **Attention Scaling Exercise**: Profile "Attention Memory Stress" config
       - Watch attention kernel time scale with S²
       - Observe memory allocation patterns in nsys
    
    3. **OOM Debugging Exercise**: Run "OOM Trigger" config
       - Capture the allocation failure in nsys
       - Identify which tensor allocation fails
       - Practice reducing batch size or enabling gradient checkpointing
    
    4. **Layer Depth Exercise**: Compare "Model A" vs "Deep Sequential"
       - Count kernel launches per training step
       - Measure kernel launch overhead accumulation
       - Observe synchronization points between layers
    
    5. **FFN Optimization Exercise**: Profile "Wide FFN" config
       - Analyze SwiGLU kernel performance
       - Compare matmul efficiency for different shapes
       - Identify memory-bound vs compute-bound operations
    """)
    return


@app.cell
def _(GPU_SPECS, go, mo, profiling_df):
    # Visualize profiling scenarios on memory scale
    _df = profiling_df.sort("peak_memory_gb")
    
    _colors = ["green" if fits else "red" for fits in _df["fits_4090"].to_list()]
    
    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        x=_df["name"].to_list(),
        y=_df["peak_memory_gb"].to_list(),
        marker_color=_colors,
        text=[f"{v:.1f} GB" for v in _df["peak_memory_gb"].to_list()],
        textposition="outside",
    ))
    _fig.add_hline(y=GPU_SPECS["vram_gb"], line_dash="dash", line_color="red",
                   annotation_text="4090 VRAM Limit (24GB)")
    
    _fig.update_layout(
        title="Profiling Scenarios: Estimated Memory Usage",
        xaxis_title="Configuration",
        yaxis_title="Peak Memory (GB)",
        height=450,
        xaxis_tickangle=-45,
    )
    
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Key Insights

    ### Model A vs Model B Comparison

    1. **Model A (Wide & Shallow)**: 
       - Fewer layers means less activation memory stacking
       - Larger d_model means more compute per token but better parallelism on GPUs
       - Lower FFN ratio may limit knowledge capacity

    2. **Model B (Narrow & Deep)**: 
       - More layers = more sequential computation, potentially lower MFU
       - Higher FFN ratio (4×) follows standard practice for knowledge storage
       - Smaller attention matrices due to smaller d_model

    ### MPS vs 4090 Key Takeaways

    1. **Performance Gap**: The 4090 achieves 17-21× higher throughput due to:
       - Tensor cores (no equivalent on Apple Silicon)
       - Higher memory bandwidth (1008 vs ~400 GB/s)
       - BF16 precision (2× memory efficiency)
    
    2. **Depth Sensitivity**: Deep models (Model B) show larger performance gaps,
       suggesting MPS has higher per-layer overhead than CUDA.
    
    3. **Development Use Case**: MPS is suitable for:
       - Code development and debugging
       - Small-scale experiments
       - Testing before deploying to GPU clusters
       - NOT for production training at scale

    ### 4090 Scaling Recommendations

    1. **Memory-limited**: With 24GB VRAM, the main constraints are:
       - Sequence length (O(S²) attention) - most severe limiter
       - Batch size × model size product

    2. **Compute-limited**: The 4090's 330 BF16 TFLOPS means:
       - Small models are memory-bandwidth bound (low MFU)
       - Larger models achieve better compute utilization
       - Sweet spot is likely 100M-500M parameter models

    3. **Practical limits**: For comfortable training with headroom:
       - Max ~200M parameters with batch=32, seq=256
       - Max ~1024 sequence length with 50M param model, batch=32
       - Use gradient checkpointing for larger configs

    ### Profiling Exercise Recommendations

    1. **Start with baselines**: Profile Model A and B to establish reference points
    2. **Stress test systematically**: Use the profiling scenarios to isolate bottlenecks
    3. **Compare across dimensions**: Vary one parameter at a time to understand scaling
    4. **Practice OOM recovery**: Intentionally trigger OOM to learn debugging techniques
    """)
    return


if __name__ == "__main__":
    app.run()
