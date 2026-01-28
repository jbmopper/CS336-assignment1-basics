import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go
    import json
    from pathlib import Path
    return Path, json, mo, pl, px


@app.cell
def _(mo):
    mo.md("""
    # Training Benchmark Analysis

    Analyzing training step throughput across different model configurations.
    """)
    return


@app.cell
def _(Path, json, pl):
    # Load benchmark data
    benchmark_file = Path("notebooks/train_benchmark_20260127_092323.json")

    with open(benchmark_file) as f:
        raw_data = json.load(f)

    # Extract metadata
    metadata = {
        "timestamp": raw_data["timestamp"],
        "device": raw_data["device"],
        "max_memory_gb": raw_data["max_memory_gb"],
        "min_run_time": raw_data["min_run_time"],
    }

    # Convert results to polars DataFrame
    df = pl.DataFrame(raw_data["results"])

    # Add computed columns
    df = df.with_columns([
        (pl.col("d_model") * pl.col("num_heads")).alias("total_params_proxy"),
        (pl.col("median_s") * 1000).alias("median_ms"),
        (pl.col("batch_size") * pl.col("seq_len")).alias("tokens_per_step"),
    ])

    print(f"Loaded {len(df)} benchmark results from {metadata['device']}")
    print(f"Columns: {df.columns}")
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Dataset Overview

    - **Total configs tested**: {len(df)}
    - **Batch sizes**: {sorted(df['batch_size'].unique().to_list())}
    - **Sequence lengths**: {sorted(df['seq_len'].unique().to_list())}
    - **d_model values**: {sorted(df['d_model'].unique().to_list())}
    - **d_head values**: {sorted(df['d_head'].unique().to_list())}
    - **num_layers values**: {sorted(df['num_layers'].unique().to_list())}
    - **d_ff values**: {sorted(df['d_ff'].unique().to_list())}
    """)
    return


@app.cell
def _(df, mo, pl):
    # Summary statistics
    summary = df.select([
        pl.col("tokens_per_sec").min().alias("min_tok/s"),
        pl.col("tokens_per_sec").max().alias("max_tok/s"),
        pl.col("tokens_per_sec").mean().alias("mean_tok/s"),
        pl.col("median_ms").min().alias("min_ms"),
        pl.col("median_ms").max().alias("max_ms"),
        pl.col("est_memory_gb").min().alias("min_mem_gb"),
        pl.col("est_memory_gb").max().alias("max_mem_gb"),
    ])

    mo.md(f"""
    ## Performance Summary

    | Metric | Min | Max | Mean |
    |--------|-----|-----|------|
    | Throughput (tok/s) | {summary['min_tok/s'][0]:,.0f} | {summary['max_tok/s'][0]:,.0f} | {summary['mean_tok/s'][0]:,.0f} |
    | Step time (ms) | {summary['min_ms'][0]:.1f} | {summary['max_ms'][0]:.1f} | - |
    | Est. memory (GB) | {summary['min_mem_gb'][0]:.2f} | {summary['max_mem_gb'][0]:.2f} | - |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Filters
    """)
    return


@app.cell
def _(df, mo):
    # Create filter controls
    batch_filter = mo.ui.dropdown(
        options=["All"] + [str(x) for x in sorted(df['batch_size'].unique().to_list())],
        value="All",
        label="Batch Size"
    )
    seq_filter = mo.ui.dropdown(
        options=["All"] + [str(x) for x in sorted(df['seq_len'].unique().to_list())],
        value="All", 
        label="Seq Length"
    )
    d_model_filter = mo.ui.dropdown(
        options=["All"] + [str(x) for x in sorted(df['d_model'].unique().to_list())],
        value="All",
        label="d_model"
    )
    d_head_filter = mo.ui.dropdown(
        options=["All"] + [str(x) for x in sorted(df['d_head'].unique().to_list())],
        value="All",
        label="d_head"
    )

    mo.hstack([batch_filter, seq_filter, d_model_filter, d_head_filter], justify="start", gap=2)
    return batch_filter, d_head_filter, d_model_filter, seq_filter


@app.cell
def _(batch_filter, d_head_filter, d_model_filter, df, pl, seq_filter):
    # Apply filters
    filtered_df = df

    if batch_filter.value != "All":
        filtered_df = filtered_df.filter(pl.col("batch_size") == int(batch_filter.value))
    if seq_filter.value != "All":
        filtered_df = filtered_df.filter(pl.col("seq_len") == int(seq_filter.value))
    if d_model_filter.value != "All":
        filtered_df = filtered_df.filter(pl.col("d_model") == int(d_model_filter.value))
    if d_head_filter.value != "All":
        filtered_df = filtered_df.filter(pl.col("d_head") == int(d_head_filter.value))

    print(f"Filtered to {len(filtered_df)} results")
    return (filtered_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Throughput vs Memory (Pareto Frontier)
    """)
    return


@app.cell
def _(filtered_df, px):
    # Throughput vs Memory scatter plot
    fig_pareto = px.scatter(
        filtered_df,
        x="est_memory_gb",
        y="tokens_per_sec",
        color="d_model",
        size="batch_size",
        hover_data=["batch_size", "seq_len", "d_model", "d_head", "num_heads", "num_layers", "d_ff", "median_ms"],
        title="Throughput vs Estimated Memory",
        labels={
            "est_memory_gb": "Estimated Memory (GB)",
            "tokens_per_sec": "Throughput (tokens/sec)",
            "d_model": "d_model"
        }
    )

    fig_pareto.update_layout(height=500)
    fig_pareto
    return


@app.cell
def _(mo):
    mo.md("""
    ## Throughput by Configuration
    """)
    return


@app.cell
def _(filtered_df, px):
    # Throughput by batch size and seq_len
    fig_batch_seq = px.scatter(
        filtered_df,
        x="batch_size",
        y="tokens_per_sec",
        color="seq_len",
        facet_col="d_model",
        hover_data=["d_head", "num_heads", "num_layers", "d_ff", "median_ms"],
        title="Throughput by Batch Size (faceted by d_model)",
        labels={
            "batch_size": "Batch Size",
            "tokens_per_sec": "Throughput (tok/s)",
            "seq_len": "Seq Length"
        }
    )

    fig_batch_seq.update_layout(height=400)
    fig_batch_seq
    return


@app.cell
def _(mo):
    mo.md("""
    ## Scaling Analysis
    """)
    return


@app.cell
def _(filtered_df, pl, px):
    # How does throughput scale with sequence length?
    seq_agg = filtered_df.group_by(["seq_len", "d_model"]).agg(
        pl.col("tokens_per_sec").mean()
    ).sort("seq_len")

    fig_seq_scaling = px.line(
        seq_agg,
        x="seq_len",
        y="tokens_per_sec",
        color="d_model",
        markers=True,
        title="Mean Throughput vs Sequence Length (expect S² degradation)",
        labels={
            "seq_len": "Sequence Length",
            "tokens_per_sec": "Mean Throughput (tok/s)",
            "d_model": "d_model"
        }
    )

    fig_seq_scaling.update_layout(height=400)
    fig_seq_scaling
    return


@app.cell
def _(filtered_df, pl, px):
    # How does throughput scale with num_layers?
    layer_agg = filtered_df.group_by(["num_layers", "d_model"]).agg(
        pl.col("tokens_per_sec").mean()
    ).sort("num_layers")

    fig_layer_scaling = px.line(
        layer_agg,
        x="num_layers",
        y="tokens_per_sec",
        color="d_model",
        markers=True,
        title="Mean Throughput vs Number of Layers",
        labels={
            "num_layers": "Number of Layers",
            "tokens_per_sec": "Mean Throughput (tok/s)",
            "d_model": "d_model"
        }
    )

    fig_layer_scaling.update_layout(height=400)
    fig_layer_scaling
    return


@app.cell
def _(mo):
    mo.md("""
    ## Top Configurations
    """)
    return


@app.cell
def _(filtered_df, mo):
    # Top 10 by throughput
    top_throughput = filtered_df.sort("tokens_per_sec", descending=True).head(10).select([
        "batch_size", "seq_len", "d_model", "d_head", "num_heads", 
        "num_layers", "d_ff", "tokens_per_sec", "median_ms", "est_memory_gb"
    ])

    mo.vstack([
        mo.md("### Top 10 by Throughput"),
        mo.ui.table(top_throughput)
    ])
    return


@app.cell
def _(filtered_df, mo, pl):
    # Top 10 by efficiency (throughput / memory)
    efficiency_df = filtered_df.with_columns([
        (pl.col("tokens_per_sec") / pl.col("est_memory_gb")).alias("efficiency")
    ])

    top_efficiency = efficiency_df.sort("efficiency", descending=True).head(10).select([
        "batch_size", "seq_len", "d_model", "d_head", "num_heads",
        "num_layers", "d_ff", "tokens_per_sec", "est_memory_gb", "efficiency"
    ])

    mo.vstack([
        mo.md("### Top 10 by Efficiency (throughput / memory)"),
        mo.ui.table(top_efficiency)
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## d_head Analysis
    """)
    return


@app.cell
def _(filtered_df, px):
    # Does d_head affect performance?
    fig_dhead = px.box(
        filtered_df,
        x="d_head",
        y="tokens_per_sec",
        color="d_model",
        title="Throughput Distribution by d_head",
        labels={
            "d_head": "Head Dimension",
            "tokens_per_sec": "Throughput (tok/s)",
            "d_model": "d_model"
        }
    )

    fig_dhead.update_layout(height=400)
    fig_dhead
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Insights

    Use the filters above to explore:
    1. **Pareto frontier**: Which configs give best throughput for a given memory budget?
    2. **S² scaling**: How badly does throughput degrade with longer sequences?
    3. **d_head sweet spot**: Is d_head=64 consistently better than 32 or 128?
    4. **Batch size scaling**: At what point do larger batches stop helping?
    """)
    return


if __name__ == "__main__":
    app.run()
