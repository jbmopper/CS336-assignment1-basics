import marimo

__generated_with = "0.19.7"
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
        value="notebooks/benchmark_results/wandb_lr_sweep_finished.parquet",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Load Training History

    Load iteration-level history to analyze training dynamics.
    You can either:
    1. **Load from parquet** - Fast, use a pre-fetched history file
    2. **Fetch from W&B** - Slower, fetches directly from W&B API

    To create a history parquet file, run:
    ```bash
    uv run notebooks/wandb_fetch_runs.py \\
      --project jbmopper-0/assignment1-basics-cs336_basics \\
      --sweep-ids <sweep_id> \\
      --include-history \\
      --history-keys "Eval Loss,Loss,LR,_step" \\
      --history-output notebooks/benchmark_results/lr_sweep_history.parquet \\
      --output notebooks/benchmark_results/lr_sweep_runs.parquet
    ```
    """)
    return


@app.cell
def _(mo):
    history_source = mo.ui.radio(
        options={"parquet": "Load from parquet file", "wandb": "Fetch from W&B API"},
        value="parquet",
        label="History source",
    )
    history_parquet_path = mo.ui.text(
        value="notebooks/benchmark_results/lr_sweep_history.parquet",
        label="History parquet file path",
    )
    load_history_btn = mo.ui.button(label="Load History")
    mo.vstack([
        history_source,
        history_parquet_path,
        load_history_btn,
    ])
    return history_parquet_path, history_source, load_history_btn


@app.cell
def _(mo):
    # W&B fetch options (only used if source is "wandb")
    wandb_project_input = mo.ui.text(
        value="jbmopper-0/assignment1-basics-cs336_basics",
        label="W&B project (entity/project)",
    )
    history_keys_input = mo.ui.text(
        value="Eval Loss,Loss,LR,_step",
        label="History keys (comma-separated)",
    )
    mo.md("**W&B API options** (only used if fetching from W&B):")
    mo.vstack([wandb_project_input, history_keys_input])
    return history_keys_input, wandb_project_input


@app.cell
def _(
    Path,
    df_raw,
    history_keys_input,
    history_parquet_path,
    history_source,
    load_history_btn,
    mo,
    pl,
    wandb_project_input,
):
    _ = load_history_btn.value
    df_history = None
    history_status = None

    if history_source.value == "parquet":
        # Load from parquet file
        path = Path(history_parquet_path.value.strip()).expanduser()
        if not path.exists():
            history_status = mo.md(f"❌ History file not found: `{path}`")
        else:
            try:
                df_history = pl.read_parquet(path)
                history_status = mo.md(
                    f"✅ Loaded history from `{path.name}` with {df_history.height} rows."
                )
            except Exception as exc:
                history_status = mo.md(f"❌ Failed to load history parquet: {exc}")
    else:
        # Fetch from W&B API
        import wandb

        mo.stop(df_raw is None, mo.md("⚠️ Load a runs parquet file first."))

        history_keys = [k.strip() for k in history_keys_input.value.split(",") if k.strip()]
        run_ids = df_raw.select("run_id").to_series().to_list()
        lr_map = dict(
            zip(
                df_raw.select("run_id").to_series().to_list(),
                df_raw.select("config.scheduler_lr_max").to_series().to_list(),
            )
        )
        run_name_map = dict(
            zip(
                df_raw.select("run_id").to_series().to_list(),
                df_raw.select("run_name").to_series().to_list(),
            )
        )

        api = wandb.Api()
        project_path = wandb_project_input.value.strip()

        history_rows = []
        for run_id in run_ids:
            try:
                run = api.run(f"{project_path}/{run_id}")
                for row in run.scan_history(keys=history_keys):
                    row["run_id"] = run_id
                    row["run_name"] = run_name_map.get(run_id, run_id)
                    row["config.scheduler_lr_max"] = lr_map.get(run_id)
                    history_rows.append(row)
            except Exception as e:
                print(f"Failed to fetch {run_id}: {e}")

        df_history = pl.from_dicts(history_rows) if history_rows else None
        history_status = (
            mo.md(f"✅ Fetched {len(history_rows)} history rows from {len(run_ids)} runs.")
            if history_rows
            else mo.md("❌ No history rows fetched.")
        )

    history_status
    return (df_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Eval Loss by Iteration

    Line chart showing eval loss over iterations, with one line per run.
    Runs are colored by learning rate to help identify which LR causes instability.
    """)
    return


@app.cell
def _(df_history, mo, pl):
    import altair as alt

    mo.stop(df_history is None, mo.md("⚠️ Load history data first."))

    # Filter to only rows with Eval Loss
    df_plot = df_history.filter(pl.col("Eval Loss").is_not_null())
    mo.stop(df_plot.height == 0, mo.md("⚠️ No Eval Loss data in history."))

    # Normalize LR column name (handle both "lr_max" and "config.scheduler_lr_max")
    lr_col = "config.scheduler_lr_max" if "config.scheduler_lr_max" in df_plot.columns else "lr_max"
    if lr_col not in df_plot.columns:
        mo.stop(True, mo.md("⚠️ No learning rate column found in history data."))

    # Rename to consistent name for plotting
    df_plot = df_plot.with_columns(pl.col(lr_col).alias("lr_max"))

    # Create a label combining run name and LR for the legend
    df_plot = df_plot.with_columns(
        pl.format("LR={} ({})", pl.col("lr_max"), pl.col("run_name")).alias("run_label")
    )

    # Sort by LR for consistent legend ordering
    df_plot = df_plot.sort("lr_max", "_step")

    chart = (
        alt.Chart(df_plot.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("_step:Q", title="Iteration"),
            y=alt.Y("Eval Loss:Q", title="Eval Loss"),
            color=alt.Color(
                "run_label:N",
                title="Run (by LR)",
                sort=alt.EncodingSortField(field="lr_max", order="ascending"),
            ),
            tooltip=["run_name", "lr_max", "_step", "Eval Loss"],
        )
        .properties(width=800, height=500, title="Eval Loss vs Iteration (by Learning Rate)")
        .interactive()
    )

    mo.ui.altair_chart(chart)
    return


if __name__ == "__main__":
    app.run()
