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


@app.cell
def _():
    def pick_first_column(columns, candidates):
        columns_set = set(columns)
        for name in candidates:
            if name in columns_set:
                return name
        return None

    def pick_first_contains(columns, substrings):
        for col in columns:
            lower = col.lower()
            if any(sub in lower for sub in substrings):
                return col
        return None
    return pick_first_column, pick_first_contains


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
def _(df_raw):
    df_raw
    return


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

    _output = None
    if sweep_selector is not None:
        _output = mo.vstack([
            mo.md("### Filter by Sweep ID"),
            sweep_selector
        ])
    else:
        _output = mo.md("⚠️ No sweep ID column found in data")

    _output
    return sweep_id_col, sweep_selector


@app.cell
def _(df_raw, mo, pl, sweep_id_col, sweep_selector):
    mo.stop(df_raw is None)

    # Filter data by selected sweeps
    df_filtered = df_raw
    if sweep_selector is not None and len(sweep_selector.value) > 0:
        df_filtered = df_raw.filter(pl.col(sweep_id_col).is_in(sweep_selector.value))

    plot_mode_selector = None

    if df_filtered.height > 0:
        # Check for required columns
        required_cols = ["_step", "Eval Loss", "run_name"]
        missing = [c for c in required_cols if c not in df_filtered.columns]

        if not missing:
            # Create plot mode selector
            plot_mode_selector = mo.ui.radio(
                options=["Line (no selection)", "Scatter with lines (selectable)"],
                value="Scatter with lines (selectable)",
                label="Plot mode"
            )

    _output = None
    if plot_mode_selector is not None:
        _output = mo.vstack([
            mo.md("### Eval Loss Over Time"),
            plot_mode_selector
        ])
    else:
        _output = mo.md("⚠️ Cannot create plot - check that data has required columns: _step, Eval Loss, run_name")

    _output
    return df_filtered, plot_mode_selector


@app.cell
def _(df_filtered, mo, plot_mode_selector, px, sweep_id_col):
    mo.stop(df_filtered is None or df_filtered.height == 0 or plot_mode_selector is None)

    # Ensure we have the LR column if available for hover
    hover_data = []
    if "config.scheduler_lr_max" in df_filtered.columns:
        hover_data.append("config.scheduler_lr_max")
    if sweep_id_col is not None and sweep_id_col in df_filtered.columns:
        hover_data.append(sweep_id_col)

    # Create the appropriate plot based on mode
    if plot_mode_selector.value == "Line (no selection)":
        # Standard line plot (not selectable)
        fig = px.line(
            df_filtered,
            x="_step",
            y="Eval Loss",
            color="run_name",
            hover_data=hover_data,
            title="Eval Loss per Iteration by Run (Line - not selectable)",
            labels={"_step": "Iteration", "Eval Loss": "Eval Loss"},
        )
        plot = mo.ui.plotly(fig)
    else:
        # Scatter plot with lines (selectable!)
        fig = px.scatter(
            df_filtered,
            x="_step",
            y="Eval Loss",
            color="run_name",
            hover_data=hover_data,
            title="Eval Loss per Iteration by Run (Scatter - click/drag to select)",
            labels={"_step": "Iteration", "Eval Loss": "Eval Loss"},
        )
        # Add lines connecting points
        fig.update_traces(mode="lines+markers", marker=dict(size=3))
        plot = mo.ui.plotly(fig)

    plot
    return (plot,)


@app.cell
def _(df_filtered, mo, plot):
    mo.stop(df_filtered is None or df_filtered.height == 0 or plot is None)

    # Display selected data from the plot
    selected_data = plot.value

    _output = None
    if selected_data is not None and len(selected_data) > 0:
        # plot.value contains the selected points as a list of dicts
        _output = mo.vstack([
            mo.md(f"### Selected Points ({len(selected_data)} points)"),
            mo.md("*Use box select or lasso select tool in the plot toolbar*"),
            mo.ui.table(selected_data, selection=None, page_size=10)
        ])
    else:
        _output = mo.md("*💡 No points selected. Use the box select (⬚) or lasso select (⚬) tool in the plot toolbar above to select data points.*")

    _output
    return


@app.cell
def _(df_filtered, mo, pick_first_column, pl):
    mo.stop(df_filtered is None or df_filtered.height == 0)

    # Create summary dataframe with max values for each run
    aggs = [
        pl.col("Eval Loss").max().alias("Max Eval Loss")
    ]

    # Add learning rate column if available
    lr_col = None
    lr_candidates = [
        "config.scheduler_lr_max",
        "config.max_lr",
        "config.lr_max",
        "max_lr",
        "lr_max",
        "LR",
        "lr",
        "learning_rate",
    ]
    lr_col = pick_first_column(df_filtered.columns, lr_candidates)
    if lr_col is not None:
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


@app.cell(hide_code=True)
def _(mo):
    import textwrap as _textwrap

    mo.md(
        _textwrap.dedent(
            """
            ## Instability diagnostics

            We consider training instability to show up as sharp positive jumps in eval loss,
            especially when those jumps happen at high learning rates. Two helpful metrics:

            - **Max Loss Jump per LR**: max over steps of `(loss_t - loss_{t-1}) / lr_t`. Large
              positive values indicate sudden loss spikes relative to the current learning rate.
            - **Max Eval Loss / Max Loss Jump**: highlights divergence or loss blow-ups.

            Compare these metrics to each run's peak learning rate to identify which peak LR
            settings are more likely to destabilize training.
            """
        ).strip()
    )
    return


@app.cell
def _(df_filtered, mo, pick_first_column, pl, px):
    mo.stop(df_filtered is None or df_filtered.height == 0)

    _columns = df_filtered.columns
    step_col = pick_first_column(_columns, ["_step", "step", "global_step"])
    loss_col = pick_first_column(
        _columns,
        ["Eval Loss", "eval_loss", "eval/loss", "Eval/Loss"],
    )
    _lr_col = pick_first_column(
        _columns,
        [
            "lr",
            "LR",
            "learning_rate",
            "train/lr",
            "optimizer_lr",
            "scheduler_lr",
            "optimizer/learning_rate",
        ],
    )
    if _lr_col is None:
        _lr_col = next(
            (
                col
                for col in _columns
                if "lr" in col.lower() and not col.lower().startswith("config.")
            ),
            None,
        )

    _missing = []
    if step_col is None:
        _missing.append("step")
    if loss_col is None:
        _missing.append("eval loss")
    if _lr_col is None:
        _missing.append("learning rate")

    if _missing:
        mo.stop(
            True,
            mo.md(
                f"⚠️ Cannot compute instability metrics (missing {', '.join(_missing)} columns)."
            ),
        )

    _max_lr_col = pick_first_column(
        _columns,
        [
            "config.scheduler_lr_max",
            "config.max_lr",
            "config.lr_max",
            "max_lr",
            "lr_max",
        ],
    )
    if _max_lr_col is None:
        _max_lr_col = _lr_col

    select_cols = ["run_name", step_col, loss_col, _lr_col]
    if _max_lr_col not in select_cols:
        select_cols.append(_max_lr_col)

    df_metric = (
        df_filtered
        .select(select_cols)
        .sort(["run_name", step_col])
        .with_columns(
            pl.col(loss_col).diff().over("run_name").alias("loss_delta")
        )
        .with_columns(
            pl.when(pl.col(_lr_col) != 0)
            .then(pl.col("loss_delta") / pl.col(_lr_col))
            .otherwise(None)
            .alias("loss_delta_over_lr")
        )
    )

    _aggs = [
        pl.col("loss_delta_over_lr").max().alias("Max Loss Jump per LR"),
        pl.col(loss_col).max().alias("Max Eval Loss"),
        pl.col("loss_delta").max().alias("Max Loss Jump"),
        pl.col(_max_lr_col).max().alias("Max LR Observed"),
    ]

    instability_df = (
        df_metric
        .group_by("run_name")
        .agg(_aggs)
        .sort("Max Loss Jump per LR", descending=True)
    )

    instability_plot = None
    if "Max LR Observed" in instability_df.columns:
        _fig = px.scatter(
            instability_df,
            x="Max LR Observed",
            y="Max Loss Jump per LR",
            hover_data=["run_name", "Max Eval Loss"],
            title="Instability vs Peak Learning Rate",
            labels={
                "Max LR Observed": "Peak LR",
                "Max Loss Jump per LR": "Max Loss Jump per LR",
            },
        )
        _fig.update_traces(marker=dict(size=7))
        instability_plot = mo.ui.plotly(_fig)

    mo.vstack(
        [
            mo.md(
                "### Instability metrics (per run)\n"
                f"- Loss column: `{loss_col}`\n"
                f"- LR column: `{_lr_col}`\n"
                f"- Peak LR column: `{_max_lr_col}`"
            ),
            mo.ui.table(instability_df, page_size=10),
            instability_plot if instability_plot is not None else mo.md(""),
        ]
    )
    return


@app.cell
def _(df_filtered, mo, pick_first_column, pick_first_contains, pl, px):
    mo.stop(df_filtered is None or df_filtered.height == 0)

    _columns = df_filtered.columns
    scheduler_col = pick_first_column(
        _columns,
        [
            "config.scheduler",
            "config.scheduler_name",
            "config.scheduler_type",
            "scheduler",
            "scheduler_name",
        ],
    )
    if scheduler_col is None:
        scheduler_col = pick_first_contains(_columns, ["scheduler"])

    if scheduler_col is None:
        mo.stop(True, mo.md("⚠️ No scheduler column found for Bayesian scheduler analysis."))

    _max_lr_col = pick_first_column(
        _columns,
        [
            "config.scheduler_lr_max",
            "config.max_lr",
            "config.lr_max",
            "max_lr",
            "lr_max",
            "lr",
            "LR",
            "learning_rate",
            "train/lr",
            "optimizer_lr",
            "scheduler_lr",
        ],
    )

    if _max_lr_col is None:
        mo.stop(
            True,
            mo.md("⚠️ No learning-rate column found for Bayesian scheduler analysis."),
        )

    df_bayes = df_filtered.filter(
        pl.col(scheduler_col)
        .cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .str.contains("bayes")
    )

    if df_bayes.height == 0:
        mo.stop(
            True,
            mo.md("⚠️ No runs tagged with a Bayesian scheduler in this parquet file."),
        )

    bayes_summary = (
        df_bayes
        .group_by("run_name")
        .agg(
            [
                pl.col(_max_lr_col).max().alias("max_lr"),
                pl.col(scheduler_col).first().alias("scheduler"),
            ]
        )
        .sort("run_name")
        .with_row_count("run_number", offset=1)
    )

    _fig = px.line(
        bayes_summary,
        x="run_number",
        y="max_lr",
        markers=True,
        hover_data=["run_name", "scheduler"],
        title="Bayesian scheduler: Max LR by run number",
        labels={
            "run_number": "Run number (sorted by name)",
            "max_lr": "Max LR",
        },
    )
    _fig.update_traces(marker=dict(size=6))
    _plot = mo.ui.plotly(_fig)

    mo.vstack(
        [
            mo.md(
                "### Bayesian scheduler behavior\n"
                f"- Scheduler column: `{scheduler_col}`\n"
                f"- Max LR column: `{_max_lr_col}`"
            ),
            _plot,
            mo.ui.table(bayes_summary, page_size=10),
        ]
    )
    return


@app.cell
def _(df_filtered):
    df_filtered
    return


if __name__ == "__main__":
    app.run()
