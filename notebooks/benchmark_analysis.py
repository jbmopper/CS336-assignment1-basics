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
        # Total parameter count (vocab_size=10000 is fixed in benchmark)
        if {"d_model", "num_layers", "d_ff"}.issubset(df.columns):
            vocab_size = 10000
            computed_cols.append(
                (
                    2 * vocab_size * pl.col("d_model") +  # embeddings + LM head
                    pl.col("d_model") +  # final RMSNorm
                    pl.col("num_layers") * (
                        2 * pl.col("d_model") +  # 2 RMSNorms per layer
                        4 * pl.col("d_model") * pl.col("d_model") +  # Q, K, V, O
                        3 * pl.col("d_model") * pl.col("d_ff")  # SwiGLU FFN
                    )
                ).alias("num_params")
            )
            # Also add millions for readability
            computed_cols.append(
                (
                    (
                        2 * vocab_size * pl.col("d_model") +
                        pl.col("d_model") +
                        pl.col("num_layers") * (
                            2 * pl.col("d_model") +
                            4 * pl.col("d_model") * pl.col("d_model") +
                            3 * pl.col("d_model") * pl.col("d_ff")
                        )
                    ) / 1e6
                ).alias("num_params_M")
            )
            # FFN expansion ratio (important architectural metric)
            computed_cols.append(
                (pl.col("d_ff") / pl.col("d_model")).alias("ffn_ratio")
            )
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

    param_range = ""
    if "num_params_M" in df.columns:
        param_range = f"- **Parameter range**: {df['num_params_M'].min():.1f}M – {df['num_params_M'].max():.1f}M"

    mo.md(f"""
    ## Dataset Overview

    - **Benchmark kind**: {metadata.get("benchmark_kind", "N/A")}
    - **Device**: {metadata.get("device", "N/A")}
    - **Timestamp**: {metadata.get("timestamp", "N/A")}
    - **Total configs tested**: {len(df)}
    {param_range}
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

    # Range sliders for continuous metrics
    if df.is_empty() or "num_params_M" not in df.columns:
        params_range_filter = mo.ui.range_slider(
            start=0, stop=100, step=1, value=[0, 100], label="Parameters (M)"
        )
    else:
        _min_p = float(df["num_params_M"].min())
        _max_p = float(df["num_params_M"].max())
        _step_p = max(0.1, (_max_p - _min_p) / 100)
        params_range_filter = mo.ui.range_slider(
            start=_min_p, stop=_max_p, step=_step_p,
            value=[_min_p, _max_p], label="Parameters (M)"
        )

    if df.is_empty() or "ffn_ratio" not in df.columns:
        ffn_ratio_filter = mo.ui.range_slider(
            start=0, stop=10, step=0.1, value=[0, 10], label="FFN Ratio"
        )
    else:
        _min_f = float(df["ffn_ratio"].min())
        _max_f = float(df["ffn_ratio"].max())
        _step_f = max(0.1, (_max_f - _min_f) / 50)
        ffn_ratio_filter = mo.ui.range_slider(
            start=_min_f, stop=_max_f, step=_step_f,
            value=[_min_f, _max_f], label="FFN Ratio"
        )

    mo.vstack([
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
        ),
        mo.hstack(
            [params_range_filter, ffn_ratio_filter],
            justify="start",
            gap=4,
        ),
    ])
    return (
        batch_filter,
        d_ff_filter,
        d_head_filter,
        d_model_filter,
        ffn_ratio_filter,
        num_heads_filter,
        num_layers_filter,
        params_range_filter,
        seq_filter,
    )


@app.cell
def _(
    batch_filter,
    d_ff_filter,
    d_head_filter,
    d_model_filter,
    df,
    ffn_ratio_filter,
    num_heads_filter,
    num_layers_filter,
    params_range_filter,
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
    # Range filters for parameters and FFN ratio
    if "num_params_M" in df.columns and params_range_filter.value:
        _p_min, _p_max = params_range_filter.value
        filtered_df = filtered_df.filter(
            (pl.col("num_params_M") >= _p_min) & (pl.col("num_params_M") <= _p_max)
        )
    if "ffn_ratio" in df.columns and ffn_ratio_filter.value:
        _f_min, _f_max = ffn_ratio_filter.value
        filtered_df = filtered_df.filter(
            (pl.col("ffn_ratio") >= _f_min) & (pl.col("ffn_ratio") <= _f_max)
        )
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

    _hover_cols = [
        "batch_size",
        "seq_len",
        "d_model",
        "d_head",
        "num_heads",
        "num_layers",
        "d_ff",
        "median_ms",
    ]
    if "num_params_M" in filtered_df.columns:
        _hover_cols.append("num_params_M")
    if "ffn_ratio" in filtered_df.columns:
        _hover_cols.append("ffn_ratio")

    _fig = px.scatter(
        filtered_df,
        x="est_memory_gb",
        y="tokens_per_sec",
        color="d_model",
        size="batch_size",
        hover_data=_hover_cols,
        title="Throughput vs Estimated Memory (click/box-select points to compare)",
        labels={
            "est_memory_gb": "Estimated Memory (GB)",
            "tokens_per_sec": "Throughput (tokens/sec)",
            "d_model": "d_model",
            "num_params_M": "Parameters (M)",
            "ffn_ratio": "FFN Ratio",
        },
    )
    _fig.update_layout(
        height=500,
        dragmode="select",  # Enable box selection by default
    )
    # Wrap in mo.ui.plotly for interactivity
    throughput_chart = mo.ui.plotly(_fig)
    throughput_chart
    return (throughput_chart,)


@app.cell
def _(filtered_df, mo, pl, throughput_chart):
    # Get selected points from the chart
    # mo.ui.plotly returns actual data values, not indices
    _raw_value = throughput_chart.value
    _output = None
    
    if not _raw_value or not isinstance(_raw_value, list) or len(_raw_value) == 0:
        _output = mo.md("*Box-select points on the chart above to compare configurations.*")
    elif filtered_df.is_empty():
        _output = mo.md("*No data available.*")
    else:
        # Selection contains actual data values - match by unique config columns
        # Map from plotly labels back to dataframe column names
        _label_to_col = {
            "Estimated Memory (GB)": "est_memory_gb",
            "Throughput (tokens/sec)": "tokens_per_sec",
            "Parameters (M)": "num_params_M",
            "FFN Ratio": "ffn_ratio",
        }
        
        # Build filter to find matching rows
        # Use config columns that uniquely identify each row
        _match_cols = ["batch_size", "seq_len", "d_model", "d_head", "num_heads", "num_layers", "d_ff"]
        
        _matched_df = filtered_df.clone()
        _match_exprs = []
        
        for point in _raw_value:
            # Build an expression that matches this specific point
            _point_conditions = []
            for col in _match_cols:
                if col in point and col in filtered_df.columns:
                    _point_conditions.append(pl.col(col) == point[col])
            
            if _point_conditions:
                # Combine all conditions for this point with AND
                _expr = _point_conditions[0]
                for cond in _point_conditions[1:]:
                    _expr = _expr & cond
                _match_exprs.append(_expr)
        
        if _match_exprs:
            # Combine all point matches with OR
            _combined = _match_exprs[0]
            for expr in _match_exprs[1:]:
                _combined = _combined | expr
            
            _selected_df = filtered_df.filter(_combined)
            
            # Select display columns
            _compare_cols = [
                "batch_size", "seq_len", "d_model", "d_head", "num_heads",
                "num_layers", "d_ff", "tokens_per_sec", "median_ms", "est_memory_gb",
            ]
            if "num_params_M" in filtered_df.columns:
                _compare_cols.append("num_params_M")
            if "ffn_ratio" in filtered_df.columns:
                _compare_cols.append("ffn_ratio")
            
            _selected_df = _selected_df.select([c for c in _compare_cols if c in _selected_df.columns])
            
            _output = mo.vstack([
                mo.md(f"### Selected Configurations ({len(_selected_df)} points)"),
                mo.ui.table(_selected_df),
            ])
        else:
            _output = mo.md("*Could not match selected points to data.*")
    
    _output
    return


@app.cell
def _(mo):
    # Calculate parameter counts for the two models
    _vocab_size = 10000
    def _calc_params(d_model, num_layers, d_ff):
        return (
            2 * _vocab_size * d_model +  # embeddings + LM head
            d_model +  # final RMSNorm
            num_layers * (
                2 * d_model +  # 2 RMSNorms per layer
                4 * d_model * d_model +  # Q, K, V, O
                3 * d_model * d_ff  # SwiGLU FFN
            )
        )

    model_a_params = _calc_params(640, 10, 1024) / 1e6
    model_b_params = _calc_params(384, 12, 1728) / 1e6

    mo.md(f"""
    ## Model selection

    To explore the properties of models with similar training characteristics but different architectures, two models that were closely matched for token throughput and memory consumption were selected:

    | Aspect | Model A | Model B |
    |--------|---------|---------|
    | batch_size | 64 | 48 |
    | seq_len | 256 | 256 |
    | d_model | 640 | 384 |
    | d_head | 64 | 32 |
    | num_heads | 10 | 12 |
    | num_layers | 10 | 12 |
    | d_ff | 1024 | 1728 |
    | **FFN ratio** | **1.6×** | **4.5×** |
    | **Parameters** | **{model_a_params:.1f}M** | **{model_b_params:.1f}M** |

    Both used ~16GB RAM and achieved 3800-3900 tokens/s throughput.

    **Key insight**: Model A has ~27% more parameters but a very low FFN expansion ratio (1.6× vs standard 4×).
    FFN layers store learned knowledge, so Model A may underperform despite its parameter advantage.
    """)
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

    _top_cols = [
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
    if "num_params_M" in filtered_df.columns:
        _top_cols.append("num_params_M")
    if "ffn_ratio" in filtered_df.columns:
        _top_cols.append("ffn_ratio")

    top_throughput = filtered_df.sort("tokens_per_sec", descending=True).head(10).select(_top_cols)

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
    _eff_cols = [
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
    if "num_params_M" in efficiency_df.columns:
        _eff_cols.append("num_params_M")
    if "ffn_ratio" in efficiency_df.columns:
        _eff_cols.append("ffn_ratio")

    top_efficiency = efficiency_df.sort("efficiency", descending=True).head(10).select(_eff_cols)

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
