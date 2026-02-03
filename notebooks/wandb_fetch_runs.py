#!/usr/bin/env python3
"""
Fetch run data from Weights & Biases and save to a local file.

Examples:
  # Fetch run summaries for specific sweeps
  python notebooks/wandb_fetch_runs.py \
    --project jbmopper-0/assignment1-basics-cs336_basics \
    --sweep-ids itfiwcnz,suxigz6c \
    --max-runs 500 \
    --output wandb_runs.parquet

  # Fetch all runs in a project
  python notebooks/wandb_fetch_runs.py \
    --project jbmopper-0/assignment1-basics-cs336_basics \
    --all-runs \
    --max-runs 1000 \
    --output wandb_runs.csv

  # Fetch iteration-level history for LR sweep analysis (includes config values)
  uv run notebooks/wandb_fetch_runs.py \
    --project jbmopper-0/assignment1-basics-cs336_basics \
    --sweep-ids suxigz6c \
    --include-history \
    --history-keys "Eval Loss,Loss,LR,_step" \
    --history-output notebooks/benchmark_results/lr_sweep_history.parquet \
    --output notebooks/benchmark_results/lr_sweep_runs.parquet

  # Fetch all runs with history
  uv run notebooks/wandb_fetch_runs.py \
    --project jbmopper-0/assignment1-basics-cs336_basics \
    --all-runs \
    --max-runs 500 \
    --output wandb_runs.parquet

  uv run notebooks/wandb_fetch_runs.py \
    --project jbmopper-0/cs336-a1-gpu-model_comp \
    --all-runs \
    --max-runs 1000 \
    --output gpu_initial.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import wandb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull W&B run data into a local file.")
    parser.add_argument(
        "--project",
        required=True,
        help="W&B project path in the form 'entity/project'.",
    )
    parser.add_argument(
        "--sweep-ids",
        default="",
        help="Comma-separated sweep IDs to fetch. If set, limits runs to these sweeps.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Fetch all runs in the project (ignores --sweep-ids).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1000,
        help="Max runs to fetch (0 means no limit). For sweeps, this is per sweep.",
    )
    parser.add_argument(
        "--states",
        default="",
        help=(
            "Comma-separated run states to include (e.g. finished,failed). "
            "If omitted, all states are included."
        ),
    )
    parser.add_argument(
        "--output",
        default="wandb_runs.parquet",
        help="Output file path (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "jsonl"],
        default=None,
        help="Optional output format override (otherwise inferred from file extension).",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Fetch all logged history rows for each run (can be large).",
    )
    parser.add_argument(
        "--history-output",
        default="wandb_history.parquet",
        help="Output path for history rows (.parquet, .csv, or .jsonl).",
    )
    parser.add_argument(
        "--history-include-config",
        action="store_true",
        default=True,
        help="Include run config values in each history row (default: True).",
    )
    parser.add_argument(
        "--history-keys",
        default="",
        help="Comma-separated history keys to fetch (default: all keys).",
    )
    parser.add_argument(
        "--history-max-rows",
        type=int,
        default=0,
        help="Max history rows per run (0 means no limit).",
    )
    return parser.parse_args()


def _infer_format(output_path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        return explicit_format
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    return "parquet"


def _max_rows_limit(value: int) -> int | None:
    return None if value <= 0 else value


def _parse_states(value: str) -> set[str] | None:
    states = {state.strip().lower() for state in value.split(",") if state.strip()}
    return states or None


def _run_to_row(run: wandb.apis.public.Run, sweep_id_override: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "sweep_id": sweep_id_override or (run.sweep.id if run.sweep else None),
    }
    row.update({f"config.{k}": v for k, v in run.config.items() if not k.startswith("_")})
    summary = getattr(run.summary, "_json_dict", dict(run.summary))
    row.update({k: v for k, v in summary.items() if not k.startswith("_")})
    return row


def _collect_history_rows(
    run: wandb.apis.public.Run,
    sweep_id_override: str | None,
    history_keys: list[str] | None,
    history_max_rows: int | None,
    include_config: bool = True,
) -> list[dict[str, Any]]:
    """Collect history rows for a run, including metadata and optionally config."""
    rows = []
    base_row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "sweep_id": sweep_id_override or (run.sweep.id if run.sweep else None),
    }
    if include_config:
        base_row.update({f"config.{k}": v for k, v in run.config.items() if not k.startswith("_")})

    count = 0
    for row in run.scan_history(keys=history_keys):
        row_out = base_row.copy()
        row_out.update(row)
        rows.append(row_out)
        count += 1
        if history_max_rows is not None and count >= history_max_rows:
            break
    return rows


def _write_history(
    history_rows: list[dict[str, Any]],
    output_path: Path,
    output_format: str,
) -> None:
    """Write history rows to file in the specified format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in history_rows:
                f.write(json.dumps(row, default=str))
                f.write("\n")
        return

    df = pl.from_dicts(history_rows, infer_schema_length=None)
    if output_format == "csv":
        df.write_csv(output_path)
    else:
        df.write_parquet(output_path)


def _fetch_project_runs(
    api: wandb.Api,
    project_path: str,
    max_runs: int,
    states: set[str] | None,
    collect_history: bool = False,
    history_keys: list[str] | None = None,
    history_max_rows: int | None = None,
    history_include_config: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch runs from a project, optionally collecting history."""
    runs = api.runs(project_path)
    limit = _max_rows_limit(max_runs)
    count = 0
    run_rows = []
    history_rows = []
    for run in runs:
        if states is not None and run.state.lower() not in states:
            continue
        if limit is not None and count >= limit:
            break
        run_rows.append(_run_to_row(run))
        if collect_history:
            history_rows.extend(
                _collect_history_rows(
                    run,
                    sweep_id_override=None,
                    history_keys=history_keys,
                    history_max_rows=history_max_rows,
                    include_config=history_include_config,
                )
            )
        count += 1
        print(f"  Fetched run {count}/{limit or '∞'}: {run.name}")
    return run_rows, history_rows


def _fetch_sweep_runs(
    api: wandb.Api,
    project_path: str,
    sweep_ids: list[str],
    max_runs: int,
    states: set[str] | None,
    collect_history: bool = False,
    history_keys: list[str] | None = None,
    history_max_rows: int | None = None,
    history_include_config: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch runs from sweeps, optionally collecting history."""
    run_rows = []
    history_rows = []
    for sweep_id in sweep_ids:
        print(f"Fetching sweep: {sweep_id}")
        sweep = api.sweep(f"{project_path}/{sweep_id}")
        limit = _max_rows_limit(max_runs)
        count = 0
        for run in sweep.runs:
            if states is not None and run.state.lower() not in states:
                continue
            if limit is not None and count >= limit:
                break
            run_rows.append(_run_to_row(run, sweep_id_override=sweep_id))
            if collect_history:
                history_rows.extend(
                    _collect_history_rows(
                        run,
                        sweep_id_override=sweep_id,
                        history_keys=history_keys,
                        history_max_rows=history_max_rows,
                        include_config=history_include_config,
                    )
                )
            count += 1
            print(f"  Fetched run {count}/{limit or '∞'}: {run.name}")
    return run_rows, history_rows


def _write_rows(rows: list[dict[str, Any]], output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, default=str))
                f.write("\n")
        return

    df = pl.from_dicts(rows, infer_schema_length=None)
    if output_format == "csv":
        df.write_csv(output_path)
    else:
        df.write_parquet(output_path)


def main() -> None:
    args = _parse_args()
    project_path = args.project.strip()
    if "/" not in project_path:
        raise SystemExit("Project must be in the form 'entity/project'.")

    sweep_ids = [s.strip() for s in args.sweep_ids.split(",") if s.strip()]
    if not args.all_runs and not sweep_ids:
        raise SystemExit("Specify --all-runs or provide --sweep-ids.")

    output_path = Path(args.output)
    output_format = _infer_format(output_path, args.format)
    history_path = Path(args.history_output)
    history_format = _infer_format(history_path, None)
    history_keys = [k.strip() for k in args.history_keys.split(",") if k.strip()]
    history_keys = history_keys or None
    history_max_rows = _max_rows_limit(args.history_max_rows)
    history_include_config = args.history_include_config
    states = _parse_states(args.states)

    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    if args.all_runs:
        print(f"Fetching all runs from {project_path}...")
        rows, history_rows = _fetch_project_runs(
            api,
            project_path,
            args.max_runs,
            states,
            collect_history=args.include_history,
            history_keys=history_keys,
            history_max_rows=history_max_rows,
            history_include_config=history_include_config,
        )
    else:
        rows, history_rows = _fetch_sweep_runs(
            api,
            project_path,
            sweep_ids,
            args.max_runs,
            states,
            collect_history=args.include_history,
            history_keys=history_keys,
            history_max_rows=history_max_rows,
            history_include_config=history_include_config,
        )

    if not rows:
        raise SystemExit("No runs found for the given query.")

    _write_rows(rows, output_path, output_format)
    print(f"Saved {len(rows)} runs to {output_path} ({output_format}).")

    if args.include_history and history_rows:
        _write_history(history_rows, history_path, history_format)
        print(f"Saved {len(history_rows)} history rows to {history_path} ({history_format}).")


if __name__ == "__main__":
    main()
