import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import wandb
    from pathlib import Path
    return Path, mo, px, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# W&B Sweep Analysis

Load sweep runs into a dataframe, save to CSV, and explore results.

**Notes**
- Make sure you are logged in (`wandb login`) or have `WANDB_API_KEY` set.
- The sweep path looks like: `entity/project/sweep_id`
"""
    )
    return


@app.cell
def _(mo):
    sweep_path = mo.ui.text(
        value="",
        label="Sweep path (entity/project/sweep_id)",
        placeholder="e.g. jbmopper-0/cs336-a1-sweep/abc123",
    )
    metric_key = mo.ui.text(value="Eval Loss", label="Metric key")
    csv_path = mo.ui.text(value="notebooks/sweep_runs.csv", label="CSV output path")
    max_runs = mo.ui.slider(1, 500, value=200, step=1, label="Max runs to load")
    log_x = mo.ui.checkbox(value=True, label="Log-scale LR axis")
    load_history = mo.ui.checkbox(value=False, label="Load per-run history")

    mo.vstack(
        [
            sweep_path,
            mo.hstack([metric_key, csv_path], justify="start", gap=2),
            mo.hstack([max_runs, log_x, load_history], justify="start", gap=2),
        ],
        gap=2,
    )
    return csv_path, load_history, log_x, max_runs, metric_key, sweep_path


@app.cell
def _(max_runs, mo, sweep_path, wandb):
    def _safe_value(value):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    sweep_data = None
    runs = None
    sweep = None
    error_msg = None

    if sweep_path.value.strip():
        try:
            api = wandb.Api()
            sweep = api.sweep(sweep_path.value.strip())
            runs = list(sweep.runs)[: max_runs.value]
            _rows = []
            for _run in runs:
                _row = {
                    "run_id": _run.id,
                    "run_name": _run.name,
                    "state": _run.state,
                    "url": _run.url,
                    "created_at": _run.created_at,
                    "runtime": _run.runtime,
                }
                for key, value in _run.config.items():
                    if key.startswith("_"):
                        continue
                    _row[f"config.{key}"] = _safe_value(value)
                _summary = getattr(_run.summary, "_json_dict", dict(_run.summary))
                for key, value in _summary.items():
                    _row[key] = _safe_value(value)
                _rows.append(_row)
            sweep_data = _rows
        except Exception as exc:
            error_msg = mo.md(f"**Error loading sweep:** `{exc}`")

    return error_msg, runs, sweep, sweep_data


@app.cell
def _(Path, csv_path, mo, sweep_data):
    import csv

    save_msg = None
    if sweep_data and len(sweep_data) > 0:
        csv_out = Path(csv_path.value).expanduser()
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all unique keys from all rows
        _all_keys = set()
        for _row in sweep_data:
            _all_keys.update(_row.keys())
        _fieldnames = sorted(_all_keys)
        
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_fieldnames)
            writer.writeheader()
            writer.writerows(sweep_data)
        
        save_msg = mo.md(f"Saved {len(sweep_data)} runs to `{csv_out}`")
    elif sweep_data is not None:
        save_msg = mo.md("No runs loaded yet.")

    return save_msg,


@app.cell
def _(metric_key, sweep_data):
    metric_col = None
    lr_col = None

    if sweep_data and len(sweep_data) > 0:
        # Get all keys from all rows
        _all_keys = set()
        for _row in sweep_data:
            _all_keys.update(_row.keys())
        
        metric_col = metric_key.value
        if metric_col not in _all_keys:
            for candidate in ("Eval Loss", "Eval/Loss", "eval_loss", "eval/loss"):
                if candidate in _all_keys:
                    metric_col = candidate
                    break
        
        if metric_col not in _all_keys:
            metric_col = None
        
        for candidate in ("config.scheduler_lr_max", "scheduler_lr_max"):
            if candidate in _all_keys:
                lr_col = candidate
                break

    return lr_col, metric_col


@app.cell
def _(lr_col, metric_col, mo, sweep_data):
    summary_display = None

    if sweep_data and len(sweep_data) > 0:
        if metric_col is None:
            summary_display = mo.md("Metric column not found. Update the metric key input.")
        else:
            # Extract metric values and filter out None
            _metric_values = [_row.get(metric_col) for _row in sweep_data if _row.get(metric_col) is not None]
            
            if _metric_values:
                _best_value = min(_metric_values)
                _best_idx = next(i for i, _row in enumerate(sweep_data) if _row.get(metric_col) == _best_value)
                _best_lr = sweep_data[_best_idx].get(lr_col) if lr_col else None
                
                _sorted_values = sorted(_metric_values)
                _n = len(_sorted_values)
                _median_value = _sorted_values[_n // 2] if _n % 2 == 1 else (_sorted_values[_n // 2 - 1] + _sorted_values[_n // 2]) / 2
                _mean_value = sum(_metric_values) / len(_metric_values)
                
                summary_md = f"""
## Summary

- Runs loaded: **{len(sweep_data)}**
- Metric column: **{metric_col}**
- Best {metric_col}: **{_best_value:.4f}** (lr: **{_best_lr}**)  
- Median {metric_col}: **{_median_value:.4f}**
- Mean {metric_col}: **{_mean_value:.4f}**
"""
                summary_display = mo.md(summary_md)

    return summary_display,


@app.cell
def _(log_x, lr_col, metric_col, mo, px, sweep_data):
    scatter_plot = None

    if sweep_data and len(sweep_data) > 0 and metric_col and lr_col:
        # Filter out rows without both metric and lr values
        _plot_data = [
            _row for _row in sweep_data 
            if _row.get(metric_col) is not None and _row.get(lr_col) is not None
        ]
        
        if _plot_data:
            _fig = px.scatter(
                _plot_data,
                x=lr_col,
                y=metric_col,
                color="state" if any("state" in _row for _row in _plot_data) else None,
                hover_data=["run_id", "run_name"],
                title="Learning rate vs. evaluation loss",
            )
            if log_x.value:
                _fig.update_xaxes(type="log")
            scatter_plot = mo.plotly(_fig)

    return scatter_plot,


@app.cell
def _(metric_col, mo, px, sweep_data):
    histogram_plot = None

    if sweep_data and len(sweep_data) > 0 and metric_col:
        # Extract metric values
        _hist_values = [
            {metric_col: _row.get(metric_col)} 
            for _row in sweep_data 
            if _row.get(metric_col) is not None
        ]
        
        if _hist_values:
            _fig = px.histogram(
                _hist_values,
                x=metric_col,
                nbins=30,
                title=f"Distribution of {metric_col}",
            )
            histogram_plot = mo.plotly(_fig)

    return histogram_plot,


@app.cell
def _(load_history, mo, runs):
    run_selector = None
    selector_display = None

    if runs and len(runs) > 0:
        options = {run.name or run.id: run.id for run in runs}
        run_selector = mo.ui.dropdown(
            options=options,
            value=next(iter(options.values())),
            label="Run for history plot",
        )
        if load_history.value:
            selector_display = mo.hstack([run_selector], justify="start")

    return run_selector, selector_display


@app.cell
def _(load_history, metric_col, mo, px, run_selector, runs):
    history_plot = None

    if load_history.value and runs and run_selector is not None:
        if metric_col is None:
            history_plot = mo.md("Metric column not found. Update the metric key input.")
        else:
            _selected_run = next((r for r in runs if r.id == run_selector.value), None)
            if _selected_run:
                _history = _selected_run.history(keys=[metric_col])
                if _history is None or len(_history) == 0:
                    history_plot = mo.md("No history found for this run.")
                else:
                    # Convert pandas DataFrame to list of dicts if needed
                    if hasattr(_history, 'to_dict'):
                        _history_records = _history.to_dict('records')
                    else:
                        _history_records = _history
                    
                    _step_col = "_step" if any("_step" in _row for _row in _history_records) else None
                    _fig = px.line(
                        _history_records,
                        x=_step_col,
                        y=metric_col,
                        title=f"{metric_col} over time ({_selected_run.name or _selected_run.id})",
                    )
                    history_plot = mo.plotly(_fig)

    return history_plot,


if __name__ == "__main__":
    app.run()
