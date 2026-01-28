import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import json
    from pathlib import Path
    return Path, json, mo, pl, px


@app.cell
def _(mo):
    mo.md("""
    # Training Benchmark Analysis

    An interactive view of training step throughput across different model
    configurations (and memory budgets), using the JSON files produced by the
    benchmarking scripts.
    """)
    return


@app.cell
def _(Path, mo):
    # File selection (keep simple: dropdown value is the path string)
    benchmark_dir = Path("notebooks")
    no_files_value = "(no files found)"
    candidates = sorted(
        list(benchmark_dir.glob("*.json"))
        + list((benchmark_dir / "benchmark_results").glob("*.json"))
    )

    if not candidates:
        mo.md("No benchmark JSON files found under `notebooks/`.")
        file_selector = mo.ui.dropdown(
            options=[no_files_value],
            value=no_files_value,
            label="Benchmark file",
        )
    else:
        options = [str(p) for p in candidates]
        file_selector = mo.ui.dropdown(
            options=options,
            value=options[-1],
            label="Benchmark file",
        )

    file_selector
    return file_selector, no_files_value


@app.cell
def _(Path, file_selector, json, no_files_value, pl):
    # Load benchmark data
    df = pl.DataFrame()
    metadata = {}

    selected_file = file_selector.value
    if selected_file not in (None, no_files_value):
        benchmark_file = Path(selected_file)
        try:
            with open(benchmark_file) as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            metadata = {"error": f"File not found: {benchmark_file}"}
            raw_data = None

        if raw_data is not None:
            # Extract metadata (best-effort; handle multiple file formats)
            if isinstance(raw_data, list):
                metadata = {
                    "benchmark_kind": "micro_benchmark",
                    "timestamp": "N/A",
                    "device": "N/A",
                    "max_memory_gb": None,
                    "min_run_time": None,
                }
                results = raw_data
            elif isinstance(raw_data, dict):
                metadata = {
                    "benchmark_kind": raw_data.get("benchmark_kind", "training_benchmark"),
                    "timestamp": raw_data.get("timestamp", "N/A"),
                    "device": raw_data.get("device", "N/A"),
                    "max_memory_gb": raw_data.get("max_memory_gb", None),
                    "min_run_time": raw_data.get("min_run_time", None),
                }
                results = raw_data.get("results", [])
            else:
                metadata = {"error": "Unsupported JSON format."}
                results = []

            df = pl.DataFrame(results)

    # Add computed columns (only if data is present)
    if len(df) > 0:
        computed_cols = []
        if {"d_model", "num_heads"}.issubset(df.columns):
            computed_cols.append((pl.col("d_model") * pl.col("num_heads")).alias("total_params_proxy"))
        if "median_s" in df.columns:
            computed_cols.append((pl.col("median_s") * 1000).alias("median_ms"))
        if {"batch_size", "seq_len"}.issubset(df.columns):
            computed_cols.append((pl.col("batch_size") * pl.col("seq_len")).alias("tokens_per_step"))
        if computed_cols:
            df = df.with_columns(computed_cols)
    return df, metadata


@app.cell
def _(df, metadata, mo):
    if metadata.get("error"):
        mo.stop(True, mo.md(f"**Error:** {metadata['error']}"))

    if df.is_empty():
        mo.stop(True, mo.md("No data loaded."))

    mo.md(f"""
    ## Dataset Overview

    - **Benchmark kind**: {metadata.get("benchmark_kind", "N/A")}
    - **Device**: {metadata.get("device", "N/A")}
    - **Timestamp**: {metadata.get("timestamp", "N/A")}
    - **Total configs tested**: {len(df)}
    - **Batch sizes**: {sorted(df['batch_size'].unique().to_list()) if 'batch_size' in df.columns else []}
    - **Sequence lengths**: {sorted(df['seq_len'].unique().to_list()) if 'seq_len' in df.columns else []}
    - **d_model values**: {sorted(df['d_model'].unique().to_list()) if 'd_model' in df.columns else []}
    - **d_head values**: {sorted(df['d_head'].unique().to_list()) if 'd_head' in df.columns else []}
    - **num_layers values**: {sorted(df['num_layers'].unique().to_list()) if 'num_layers' in df.columns else []}
    - **d_ff values**: {sorted(df['d_ff'].unique().to_list()) if 'd_ff' in df.columns else []}
    """)
    return


@app.cell
def _(df, mo):
    mo.md("## Interactive Filters")

    filterable = {"batch_size", "seq_len", "d_model", "d_head", "num_heads", "num_layers", "d_ff"}
    has_filters = filterable.issubset(df.columns)

    if df.is_empty() or not has_filters:
        batch_filter = mo.ui.multiselect(options=[], value=[], label="Batch Size")
        seq_filter = mo.ui.multiselect(options=[], value=[], label="Seq Length")
        d_model_filter = mo.ui.multiselect(options=[], value=[], label="d_model")
        d_head_filter = mo.ui.multiselect(options=[], value=[], label="d_head")
        num_heads_filter = mo.ui.multiselect(options=[], value=[], label="num_heads")
        num_layers_filter = mo.ui.multiselect(options=[], value=[], label="num_layers")
        d_ff_filter = mo.ui.multiselect(options=[], value=[], label="d_ff")
    else:
        batch_filter = mo.ui.multiselect(
            options=sorted(df["batch_size"].unique().to_list()),
            value=[],
            label="Batch Size",
        )
        seq_filter = mo.ui.multiselect(
            options=sorted(df["seq_len"].unique().to_list()),
            value=[],
            label="Seq Length",
        )
        d_model_filter = mo.ui.multiselect(
            options=sorted(df["d_model"].unique().to_list()),
            value=[],
            label="d_model",
        )
        d_head_filter = mo.ui.multiselect(
            options=sorted(df["d_head"].unique().to_list()),
            value=[],
            label="d_head",
        )
        num_heads_filter = mo.ui.multiselect(
            options=sorted(df["num_heads"].unique().to_list()),
            value=[],
            label="num_heads",
        )
        num_layers_filter = mo.ui.multiselect(
            options=sorted(df["num_layers"].unique().to_list()),
            value=[],
            label="num_layers",
        )
        d_ff_filter = mo.ui.multiselect(
            options=sorted(df["d_ff"].unique().to_list()),
            value=[],
            label="d_ff",
        )

    mo.hstack(
        [
            batch_filter,
            seq_filter,
            d_model_filter,
            d_head_filter,
            num_heads_filter,
            num_layers_filter,
            d_ff_filter,
        ],
        justify="start",
        gap=2,
        wrap=True,
    )
    return (
        batch_filter,
        d_head_filter,
        d_model_filter,
        d_ff_filter,
        num_heads_filter,
        num_layers_filter,
        seq_filter,
    )


@app.cell
def _(
    batch_filter,
    d_head_filter,
    d_model_filter,
    d_ff_filter,
    df,
    num_heads_filter,
    num_layers_filter,
    pl,
    seq_filter,
):
    filtered_df = df
    if "batch_size" in df.columns and batch_filter.value:
        filtered_df = filtered_df.filter(pl.col("batch_size").is_in(batch_filter.value))
    if "seq_len" in df.columns and seq_filter.value:
        filtered_df = filtered_df.filter(pl.col("seq_len").is_in(seq_filter.value))
    if "d_model" in df.columns and d_model_filter.value:
        filtered_df = filtered_df.filter(pl.col("d_model").is_in(d_model_filter.value))
    if "d_head" in df.columns and d_head_filter.value:
        filtered_df = filtered_df.filter(pl.col("d_head").is_in(d_head_filter.value))
    if "num_heads" in df.columns and num_heads_filter.value:
        filtered_df = filtered_df.filter(pl.col("num_heads").is_in(num_heads_filter.value))
    if "num_layers" in df.columns and num_layers_filter.value:
        filtered_df = filtered_df.filter(pl.col("num_layers").is_in(num_layers_filter.value))
    if "d_ff" in df.columns and d_ff_filter.value:
        filtered_df = filtered_df.filter(pl.col("d_ff").is_in(d_ff_filter.value))
    return (filtered_df,)


@app.cell
def _(filtered_df, mo, pl):
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to summarize."))
    _required_cols = {"tokens_per_sec", "median_ms", "est_memory_gb"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for summary: {_missing}"))

    summary = filtered_df.select(
        [
            pl.col("tokens_per_sec").min().alias("min_tok_s"),
            pl.col("tokens_per_sec").max().alias("max_tok_s"),
            pl.col("tokens_per_sec").mean().alias("mean_tok_s"),
            pl.col("median_ms").min().alias("min_ms"),
            pl.col("median_ms").max().alias("max_ms"),
            pl.col("est_memory_gb").min().alias("min_mem_gb"),
            pl.col("est_memory_gb").max().alias("max_mem_gb"),
        ]
    )

    mo.md(f"""
    ## Performance Summary (filtered)

    | Metric | Min | Max | Mean |
    |--------|-----|-----|------|
    | Throughput (tok/s) | {summary['min_tok_s'][0]:,.0f} | {summary['max_tok_s'][0]:,.0f} | {summary['mean_tok_s'][0]:,.0f} |
    | Step time (ms) | {summary['min_ms'][0]:.1f} | {summary['max_ms'][0]:.1f} | - |
    | Est. memory (GB) | {summary['min_mem_gb'][0]:.2f} | {summary['max_mem_gb'][0]:.2f} | - |
    """)
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("## Throughput vs Memory")
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to plot."))
    _required_cols = {"est_memory_gb", "tokens_per_sec", "d_model", "batch_size"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for plot: {_missing}"))

    _fig = px.scatter(
        filtered_df,
        x="est_memory_gb",
        y="tokens_per_sec",
        color="d_model",
        size="batch_size",
        hover_data=[
            "batch_size",
            "seq_len",
            "d_model",
            "d_head",
            "num_heads",
            "num_layers",
            "d_ff",
            "median_ms",
        ],
        title="Throughput vs Estimated Memory",
        labels={
            "est_memory_gb": "Estimated Memory (GB)",
            "tokens_per_sec": "Throughput (tokens/sec)",
            "d_model": "d_model",
        },
    )
    _fig.update_layout(height=500)
    _fig
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("## Throughput by Configuration")
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to plot."))
    _required_cols = {"batch_size", "tokens_per_sec", "seq_len", "d_model"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for plot: {_missing}"))

    _fig = px.scatter(
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
            "seq_len": "Seq Length",
        },
    )
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(filtered_df, mo, pl, px):
    mo.md("## Scaling Analysis")
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to plot."))
    _required_cols = {"seq_len", "d_model", "tokens_per_sec"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for plot: {_missing}"))

    seq_agg = (
        filtered_df.group_by(["seq_len", "d_model"])
        .agg(pl.col("tokens_per_sec").mean())
        .sort("seq_len")
    )
    _fig = px.line(
        seq_agg,
        x="seq_len",
        y="tokens_per_sec",
        color="d_model",
        markers=True,
        title="Mean Throughput vs Sequence Length",
        labels={"seq_len": "Sequence Length", "tokens_per_sec": "Mean Throughput (tok/s)"},
    )
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(filtered_df, mo, pl, px):
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to plot."))
    _required_cols = {"num_layers", "d_model", "tokens_per_sec"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for plot: {_missing}"))

    layer_agg = (
        filtered_df.group_by(["num_layers", "d_model"])
        .agg(pl.col("tokens_per_sec").mean())
        .sort("num_layers")
    )
    _fig = px.line(
        layer_agg,
        x="num_layers",
        y="tokens_per_sec",
        color="d_model",
        markers=True,
        title="Mean Throughput vs Number of Layers",
        labels={"num_layers": "Number of Layers", "tokens_per_sec": "Mean Throughput (tok/s)"},
    )
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(filtered_df, mo):
    mo.md("## Top Configurations")
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to show."))

    top_throughput = filtered_df.sort("tokens_per_sec", descending=True).head(10).select(
        [
            "batch_size",
            "seq_len",
            "d_model",
            "d_head",
            "num_heads",
            "num_layers",
            "d_ff",
            "tokens_per_sec",
            "median_ms",
            "est_memory_gb",
        ]
    )

    mo.vstack([mo.md("### Top 10 by Throughput"), mo.ui.table(top_throughput)])
    return


@app.cell
def _(filtered_df, mo, pl):
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to show."))
    _required_cols = {"tokens_per_sec", "est_memory_gb"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for efficiency table: {_missing}"))

    efficiency_df = filtered_df.with_columns(
        [(pl.col("tokens_per_sec") / pl.col("est_memory_gb")).alias("efficiency")]
    )
    top_efficiency = efficiency_df.sort("efficiency", descending=True).head(10).select(
        [
            "batch_size",
            "seq_len",
            "d_model",
            "d_head",
            "num_heads",
            "num_layers",
            "d_ff",
            "tokens_per_sec",
            "est_memory_gb",
            "efficiency",
        ]
    )

    mo.vstack([mo.md("### Top 10 by Efficiency (throughput / memory)"), mo.ui.table(top_efficiency)])
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("## d_head Analysis")
    if filtered_df.is_empty():
        mo.stop(True, mo.md("No data to plot."))
    _required_cols = {"d_head", "tokens_per_sec", "d_model"}
    if not _required_cols.issubset(filtered_df.columns):
        _missing = sorted(_required_cols - set(filtered_df.columns))
        mo.stop(True, mo.md(f"Missing required columns for plot: {_missing}"))

    _fig = px.box(
        filtered_df,
        x="d_head",
        y="tokens_per_sec",
        color="d_model",
        title="Throughput Distribution by d_head",
        labels={"d_head": "Head Dimension", "tokens_per_sec": "Throughput (tok/s)", "d_model": "d_model"},
    )
    _fig.update_layout(height=400)
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Insights

    Use the filters above to explore:
    1. **Pareto intuition**: which configs give best throughput for a given memory budget?
    2. **Sequence scaling**: how badly does throughput degrade with longer sequences?
    3. **d_head effect**: is d_head=64 consistently better than 32 or 128?
    4. **Batch scaling**: at what point do larger batches stop helping?
    """)
    return


if __name__ == "__main__":
    app.run()
