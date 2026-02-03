import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    return Path, mo, pl


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
        value="notebooks/benchmark_results/wandb_runs.parquet",
        label="Parquet file path",
        placeholder="notebooks/benchmark_results/wandb_runs.parquet",
    )
    reload_data = mo.ui.button(label="Load parquet")
    max_rows = mo.ui.number(
        start=10,
        stop=100000,
        step=100,
        value=2000,
        label="Max rows to display",
    )

    mo.vstack(
        [
            parquet_path,
            mo.hstack([reload_data, max_rows]),
        ]
    )
    return max_rows, parquet_path, reload_data


@app.cell
def _(Path, mo, parquet_path, pl, reload_data):
    _ = reload_data.value
    path_value = parquet_path.value.strip()
    mo.stop(not path_value)

    display = None
    df_raw = None
    path = Path(path_value).expanduser()
    if not path.exists():
        display = mo.md(f"❌ File not found: `{path}`")
    else:
        try:
            df_raw = pl.read_parquet(path)
            display = mo.md(
                f"✅ Loaded `{path.name}` with {df_raw.height} rows and {len(df_raw.columns)} columns."
            )
        except Exception as exc:
            display = mo.md(f"❌ Failed to load parquet: {exc}")
    return display, df_raw


@app.cell
def _(display, mo):
    display


@app.cell
def _(df_raw, mo):
    mo.stop(df_raw is None)

    columns_sorted = sorted(df_raw.columns)

    run_ids = (
        df_raw.select("run_id")
        .drop_nulls()
        .unique()
        .to_series()
        .sort()
        .to_list()
        if "run_id" in df_raw.columns
        else []
    )
    states = (
        df_raw.select("state")
        .drop_nulls()
        .unique()
        .to_series()
        .sort()
        .to_list()
        if "state" in df_raw.columns
        else []
    )
    sweep_ids = (
        df_raw.select("sweep_id")
        .drop_nulls()
        .unique()
        .to_series()
        .sort()
        .to_list()
        if "sweep_id" in df_raw.columns
        else []
    )

    run_filter_mode = mo.ui.radio(
        options=["All runs", "Include selected", "Exclude selected"],
        value="All runs",
        label="Run filter mode",
    )
    run_filter_ids = mo.ui.multiselect(
        options=run_ids,
        value=[],
        label="Run IDs",
    )
    state_filter_mode = mo.ui.radio(
        options=["All states", "Include selected", "Exclude selected"],
        value="All states",
        label="State filter mode",
    )
    state_filter_values = mo.ui.multiselect(
        options=states,
        value=[],
        label="States",
    )

    sweep_filter_mode = mo.ui.radio(
        options=["All sweeps", "Include selected", "Exclude selected"],
        value="All sweeps",
        label="Sweep filter mode",
    )
    sweep_filter_values = mo.ui.multiselect(
        options=sweep_ids,
        value=[],
        label="Sweep IDs",
    )

    run_name_contains = mo.ui.text(label="Run name contains", value="")
    columns_mode = mo.ui.radio(
        options=["All columns", "Selected columns"],
        value="All columns",
        label="Column selection",
    )
    default_columns = columns_sorted[:12] if len(columns_sorted) > 12 else columns_sorted
    columns_selected = mo.ui.multiselect(
        options=columns_sorted,
        value=default_columns,
        label="Visible columns",
    )

    controls = mo.vstack(
        [
            mo.hstack([run_filter_mode, state_filter_mode, sweep_filter_mode]),
            mo.hstack([run_filter_ids, state_filter_values, sweep_filter_values]),
            run_name_contains,
            mo.hstack([columns_mode, columns_selected]),
        ]
    )
    return (
        columns_mode,
        columns_selected,
        controls,
        run_filter_ids,
        run_filter_mode,
        run_name_contains,
        state_filter_mode,
        state_filter_values,
        sweep_filter_mode,
        sweep_filter_values,
    )


@app.cell
def _(
    columns_mode,
    columns_selected,
    df_raw,
    max_rows,
    mo,
    pl,
    run_filter_ids,
    run_filter_mode,
    run_name_contains,
    state_filter_mode,
    state_filter_values,
    sweep_filter_mode,
    sweep_filter_values,
):
    mo.stop(df_raw is None)
    df_filtered = df_raw

    if run_filter_mode.value != "All runs" and run_filter_ids.value:
        if run_filter_mode.value == "Include selected":
            df_filtered = df_filtered.filter(pl.col("run_id").is_in(run_filter_ids.value))
        else:
            df_filtered = df_filtered.filter(~pl.col("run_id").is_in(run_filter_ids.value))

    if state_filter_mode.value != "All states" and state_filter_values.value:
        if state_filter_mode.value == "Include selected":
            df_filtered = df_filtered.filter(pl.col("state").is_in(state_filter_values.value))
        else:
            df_filtered = df_filtered.filter(~pl.col("state").is_in(state_filter_values.value))

    if sweep_filter_mode.value != "All sweeps" and sweep_filter_values.value:
        if sweep_filter_mode.value == "Include selected":
            df_filtered = df_filtered.filter(pl.col("sweep_id").is_in(sweep_filter_values.value))
        else:
            df_filtered = df_filtered.filter(~pl.col("sweep_id").is_in(sweep_filter_values.value))

    name_filter = run_name_contains.value.strip()
    if name_filter and "run_name" in df_filtered.columns:
        df_filtered = df_filtered.filter(
            pl.col("run_name")
            .cast(pl.Utf8)
            .str.contains(name_filter, literal=True)
        )

    if columns_mode.value == "Selected columns" and columns_selected.value:
        df_filtered = df_filtered.select(columns_selected.value)

    row_limit = max_rows.value
    if row_limit and row_limit > 0:
        df_filtered = df_filtered.head(int(row_limit))
    return (df_filtered,)


@app.cell
def _(controls, mo):
    mo.md("## Filters")
    controls
    return


@app.cell
def _(df_filtered, mo):
    mo.md(f"""
    **Showing {df_filtered.height} rows × {len(df_filtered.columns)} columns**
    """)
    return


@app.cell
def _(df_filtered, mo):
    mo.ui.table(df_filtered.to_dicts())
    return


if __name__ == "__main__":
    app.run()
