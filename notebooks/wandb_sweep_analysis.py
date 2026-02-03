import marimo

__generated_with = "0.19.7"
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
    mo.md(r"""
    # W&B Sweep Analysis

    Load sweep runs into a dataframe, save to CSV, and explore results.

    **Notes**
    - You can load runs locally from the `wandb/` directory without any API access.
    - To use the W&B API, uncheck "Use local wandb/ directory" and set the sweep path.
    - The sweep path looks like: `entity/project/sweep_id`.
    """)
    return


@app.cell
def _(mo):
    sweep_path = mo.ui.text(
        value="jbmopper-0/assignment1-basics-cs336_basics",
        label="Sweep path (entity/project/sweep_id) or Project path (entity/project)",
        placeholder="e.g. jbmopper-0/assignment1-basics-cs336_basics/abc123",
    )
    use_local = mo.ui.checkbox(value=False, label="Use local wandb/ directory")
    wandb_dir = mo.ui.text(value="wandb", label="Local wandb directory")
    metric_key = mo.ui.text(value="Eval Loss", label="Metric key")
    csv_path = mo.ui.text(value="notebooks/sweep_runs.csv", label="CSV output path")
    max_runs = mo.ui.slider(1, 500, value=200, step=1, label="Max runs to load")
    log_x = mo.ui.checkbox(value=True, label="Log-scale LR axis")
    load_history = mo.ui.checkbox(value=False, label="Load per-run history")

    mo.vstack(
        [
            sweep_path,
            mo.hstack([use_local, wandb_dir], justify="start", gap=2),
            mo.hstack([metric_key, csv_path], justify="start", gap=2),
            mo.hstack([max_runs, log_x, load_history], justify="start", gap=2),
        ],
        gap=2,
    )
    return (
        csv_path,
        load_history,
        log_x,
        max_runs,
        metric_key,
        sweep_path,
        use_local,
        wandb_dir,
    )


@app.cell
def _(Path, max_runs, mo, sweep_path, use_local, wandb, wandb_dir):
    import json
    from datetime import datetime
    try:
        import yaml
    except Exception:
        yaml = None

    def _safe_value(value):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        return str(value)

    def _unwrap_config_value(value):
        if isinstance(value, dict) and "value" in value and len(value) == 1:
            return value["value"]
        return value

    def _parse_run_timestamp(run_name: str):
        parts = run_name.split("-")
        if len(parts) >= 3:
            try:
                return datetime.strptime(parts[1], "%Y%m%d_%H%M%S")
            except Exception:
                return None
        return None

    def _load_local_runs(base_dir: Path, limit: int):
        if not base_dir.exists():
            return []
        run_dirs = [
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith("run-")
        ]
        run_dirs = sorted(
            run_dirs,
            key=lambda p: _parse_run_timestamp(p.name) or datetime.min,
            reverse=True,
        )
        rows = []
        for run_dir in run_dirs[:limit]:
            files_dir = run_dir / "files"
            summary_path = files_dir / "wandb-summary.json"
            config_path = files_dir / "config.yaml"
            metadata_path = files_dir / "wandb-metadata.json"

            if not summary_path.exists() and not config_path.exists():
                continue

            summary = {}
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    summary = {}

            config = {}
            if config_path.exists() and yaml is not None:
                try:
                    config = yaml.safe_load(config_path.read_text()) or {}
                except Exception:
                    config = {}

            metadata = {}
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                except Exception:
                    metadata = {}

            run_id = run_dir.name.split("-")[-1]
            created_at = metadata.get("startedAt")
            if created_at is None:
                _ts = _parse_run_timestamp(run_dir.name)
                if _ts is not None:
                    created_at = _ts.isoformat()

            runtime = summary.get("_runtime")
            if runtime is None:
                runtime = summary.get("_wandb", {}).get("runtime")

            row = {
                "run_id": run_id,
                "run_name": run_id,
                "state": "finished" if runtime else "unknown",
                "url": None,
                "created_at": created_at,
                "runtime": runtime,
                "local_path": str(run_dir),
            }

            for key, value in config.items():
                if key.startswith("_"):
                    continue
                row[f"config.{key}"] = _safe_value(_unwrap_config_value(value))

            for key, value in summary.items():
                row[key] = _safe_value(value)

            rows.append(row)
        return rows

    sweep_data = None
    runs = None

    if use_local.value or not sweep_path.value.strip():
        base_dir = Path(wandb_dir.value).expanduser()
        sweep_data = _load_local_runs(base_dir, max_runs.value)
        if not base_dir.exists():
            mo.md(f"**Local wandb directory not found:** `{base_dir}`")
        elif yaml is None:
            mo.md("`pyyaml` is not available; config values may be missing.")
        elif not sweep_data:
            mo.md("No local runs found yet.")
        else:
            mo.md(f"Loaded {len(sweep_data)} local runs from `{base_dir}`.")
    else:
        path_str = sweep_path.value.strip().rstrip('/')
        path_parts = path_str.split("/")
        try:
            api = wandb.Api()
            
            if len(path_parts) == 2:
                # Treat as project path -> List sweeps
                entity, project = path_parts
                # api.sweeps returns a list of sweeps for the project
                # We can't iterate directly if it returns a generator or paginated list easily without knowing internal API, 
                # but usually it returns a list-like object.
                _sweeps = api.sweeps(entity=entity, project=project)
                
                if not _sweeps:
                     mo.md(f"No sweeps found in project `{path_str}`.")
                else:
                    _sweep_md = f"### Sweeps in `{path_str}`\n\n"
                    _sweep_md += "| ID | Name | State |\n"
                    _sweep_md += "|----|------|-------|\n"
                    # Limit to 20 sweeps to avoid clutter
                    for _s in list(_sweeps)[:20]:
                        # Construct full path for copy-pasting
                        _full_path = f"{entity}/{project}/{_s.id}"
                        _sweep_md += f"| `{_full_path}` | {_s.name} | {_s.state} |\n"
                    
                    _sweep_md += "\n\n**Copy a sweep path from the table above into the input field to analyze runs.**"
                    mo.md(_sweep_md)
            else:
                # Treat as sweep path
                sweep = api.sweep(path_str)
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
                mo.md(f"Loaded {len(sweep_data)} runs from sweep `{path_str}`.")
        except Exception as exc:
            mo.md(f"**Error loading via API:** `{exc}`\n\nMake sure you are logged in (`wandb login`) and the path is correct.")
    return runs, sweep_data


@app.cell
def _(Path, csv_path, mo, sweep_data):
    import csv

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

        mo.md(f"Saved {len(sweep_data)} runs to `{csv_out}`")
    elif sweep_data is not None:
        mo.md("No runs loaded yet.")
    return


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

        for candidate in (
            "config.scheduler_lr_max",
            "scheduler_lr_max",
            "config.lr",
            "lr",
            "LR",
            "learning_rate",
            "config.learning_rate",
        ):
            if candidate in _all_keys:
                lr_col = candidate
                break
    return lr_col, metric_col


@app.cell
def _(lr_col, metric_col, mo, sweep_data):
    if sweep_data and len(sweep_data) > 0:
        if metric_col is None:
            mo.md("Metric column not found. Update the metric key input.")
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
                mo.md(summary_md)
    return


@app.cell
def _(log_x, lr_col, metric_col, mo, px, sweep_data):
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
            mo.plotly(_fig)
    return


@app.cell
def _(metric_col, mo, px, sweep_data):
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
            mo.plotly(_fig)
    return


@app.cell
def _(load_history, mo, runs):
    run_selector = None

    if runs and len(runs) > 0:
        options = {run.name or run.id: run.id for run in runs}
        run_selector = mo.ui.dropdown(
            options=options,
            value=next(iter(options.values())),
            label="Run for history plot",
        )
        if load_history.value:
            mo.hstack([run_selector], justify="start")
    elif load_history.value:
        mo.md("Run history requires W&B API access; switch off local loading.")
    return (run_selector,)


@app.cell
def _(load_history, metric_col, mo, px, run_selector, runs):
    if load_history.value and runs and run_selector is not None:
        if metric_col is None:
            mo.md("Metric column not found. Update the metric key input.")
        else:
            _selected_run = next((r for r in runs if r.id == run_selector.value), None)
            if _selected_run:
                _history = _selected_run.history(keys=[metric_col])
                if _history is None or len(_history) == 0:
                    mo.md("No history found for this run.")
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
                    mo.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
