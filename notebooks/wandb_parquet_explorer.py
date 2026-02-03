import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    from pathlib import Path
    return Path, mo, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # W&B Parquet Explorer

    Load a W&B runs parquet file and interactively filter runs and columns.
    """)
    return


@app.cell
def _(mo):
    parquet_path = mo.ui.text(
        value="notebooks/benchmark_results/lr_sweep_history.parquet",
        label="Parquet file path",
        placeholder="notebooks/benchmark_results/",
    )
    reload_data = mo.ui.button(label="Load parquet")
    max_rows = mo.ui.number(
        start=0,
        stop=100000,
        step=100,
        value=5000,
        label="Max rows to display",
    )

    mo.vstack(
        [
            parquet_path,
            mo.hstack([reload_data, max_rows]),
        ]
    )
    return parquet_path, reload_data


@app.cell
def _(Path, mo, parquet_path, pl, reload_data):
    _ = reload_data.value
    path_value = parquet_path.value.strip()
    mo.stop(not path_value)

    load_status = None
    df_raw = None
    path = Path(path_value).expanduser()
    if not path.exists():
        load_status = mo.md(f"❌ File not found: `{path}`")
    else:
        try:
            df_raw = pl.read_parquet(path)
            load_status = mo.md(
                f"✅ Loaded `{path.name}` with {df_raw.height} rows and {len(df_raw.columns)} columns."
            )
        except Exception as exc:
            load_status = mo.md(f"❌ Failed to load parquet: {exc}")

    load_status
    return (df_raw,)


@app.cell
def _(df_raw, mo):
    mo.stop(df_raw is None)

    # Identify sweep ID column
    sweep_id_col = None
    for col in ["sweep_id", "sweep", "Sweep"]:
        if col in df_raw.columns:
            sweep_id_col = col
            break

    # Get unique sweep IDs if available
    sweep_selector = None
    if sweep_id_col is not None:
        unique_sweeps = sorted(df_raw[sweep_id_col].unique().to_list())
        sweep_selector = mo.ui.multiselect(
            options=unique_sweeps,
            value=unique_sweeps,  # Select all by default
            label=f"Select Sweep IDs ({len(unique_sweeps)} total)"
        )

    if sweep_selector is not None:
        mo.vstack([
            mo.md("### Filter by Sweep ID"),
            sweep_selector
        ])
    else:
        mo.md("⚠️ No sweep ID column found in data")

    return sweep_id_col, sweep_selector


@app.cell
def _(df_raw, mo, pl, px, sweep_id_col, sweep_selector):
    mo.stop(df_raw is None)

    # Filter data by selected sweeps
    df_filtered = df_raw
    if sweep_selector is not None and len(sweep_selector.value) > 0:
        df_filtered = df_raw.filter(pl.col(sweep_id_col).is_in(sweep_selector.value))

    plot = mo.md("Select a valid parquet file to view plots.")
    fig = None

    if df_filtered.height > 0:
        # Check for required columns
        required_cols = ["_step", "Eval Loss", "run_name"]
        missing = [c for c in required_cols if c not in df_filtered.columns]

        if missing:
            plot = mo.md(f"⚠️ Cannot plot: Missing columns {missing}")
        else:
            # Ensure we have the LR column if available for hover
            hover_data = []
            if "config.scheduler_lr_max" in df_filtered.columns:
                hover_data.append("config.scheduler_lr_max")
            if sweep_id_col is not None and sweep_id_col in df_filtered.columns:
                hover_data.append(sweep_id_col)

            # Plotly Express
            fig = px.line(
                df_filtered,
                x="_step",
                y="Eval Loss",
                color="run_name",
                hover_data=hover_data,
                title="Eval Loss per Iteration by Run",
                labels={"_step": "Iteration", "Eval Loss": "Eval Loss"},
            )
            plot = mo.ui.plotly(fig)
    else:
        plot = mo.md("⚠️ No data to plot with selected filters")

    plot
    return plot, fig, df_filtered


@app.cell
def _(df_filtered, mo, pl):
    mo.stop(df_filtered is None or df_filtered.height == 0)

    # Create summary dataframe with max values for each run
    aggs = [
        pl.col("Eval Loss").max().alias("Max Eval Loss")
    ]

    # Add learning rate column if available
    lr_col = None
    if "config.scheduler_lr_max" in df_filtered.columns:
        lr_col = "config.scheduler_lr_max"
        aggs.append(pl.col(lr_col).max().alias("Max Learning Rate"))
    elif "LR" in df_filtered.columns:
        lr_col = "LR"
        aggs.append(pl.col(lr_col).max().alias("Max Learning Rate"))

    summary_df = (
        df_filtered
        .group_by("run_name")
        .agg(aggs)
        .sort("Max Eval Loss", descending=True)
    )

    mo.vstack([
        mo.md(f"### Summary of {summary_df.height} runs (reduced along iteration)"),
        mo.md(f"Showing max eval loss and max learning rate for each run"),
        summary_df
    ])
    return


@app.cell
def _(df_filtered):
    df_filtered
    return


if __name__ == "__main__":
    app.run()
