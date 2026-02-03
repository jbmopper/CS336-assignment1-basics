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
def _(df_raw, mo, px):
    mo.stop(df_raw is None)

    plot = mo.md("Select a valid parquet file to view plots.")
    if df_raw is not None:
        # Check for required columns
        required_cols = ["_step", "Eval Loss", "run_name"]
        missing = [c for c in required_cols if c not in df_raw.columns]

        if missing:
            plot = mo.md(f"⚠️ Cannot plot: Missing columns {missing}")
        else:
            # Convert to pandas for plotly
            # Ensure we have the LR column if available for hover
            hover_data = []
            if "config.scheduler_lr_max" in df_raw.columns:
                hover_data.append("config.scheduler_lr_max")

            # Plotly Express
            fig = px.line(
                df_raw,
                x="_step",
                y="Eval Loss",
                color="run_name",
                hover_data=hover_data,
                title="Eval Loss per Iteration by Run",
                labels={"_step": "Iteration", "Eval Loss": "Eval Loss"},
            )
            plot = mo.ui.plotly(fig)
    else:
        fig = None

    plot
    return plot, fig


@app.cell
def _(df_raw, fig, mo, pl, plot):
    mo.stop(df_raw is None)
    mo.stop(fig is None)

    selected_runs_summary = None
    if plot.value and "points" in plot.value and len(plot.value["points"]) > 0:
        # Extract run names from selected points
        selected_curve_indices = {p["curveNumber"] for p in plot.value["points"]}
        selected_run_names = {fig.data[i].name for i in selected_curve_indices}
        
        if selected_run_names:
            # Filter
            filtered_df = df_raw.filter(pl.col("run_name").is_in(selected_run_names))
            
            # Aggregations
            aggs = [
                pl.col("Eval Loss").max().alias("Max Eval Loss")
            ]
            
            if "config.scheduler_lr_max" in df_raw.columns:
                aggs.append(pl.col("config.scheduler_lr_max").max().alias("Max LR"))
            elif "LR" in df_raw.columns:
                aggs.append(pl.col("LR").max().alias("Max LR"))

            selected_runs_summary = (
                filtered_df.group_by("run_name")
                .agg(aggs)
                .sort("Max Eval Loss", descending=True)
            )
    
    # Display the filtered dataframe or message
    if selected_runs_summary is not None:
        output = mo.vstack([
            mo.md(f"### Summary for {selected_runs_summary.height} selected runs"),
            selected_runs_summary
        ])
    else:
        output = mo.md("Select traces on the plot (box/lasso select) to view run summary.")
        
    output
    return (output, selected_runs_summary)


@app.cell
def _(df_raw):
    df_raw
    return


if __name__ == "__main__":
    app.run()
