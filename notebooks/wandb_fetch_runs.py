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
import math
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
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
        help=(
            "Comma-separated history keys to keep in output. "
            "History is scanned without server-side key filtering to avoid sparse-row drops."
        ),
    )
    parser.add_argument(
        "--history-max-rows",
        type=int,
        default=0,
        help="Max history rows per run (0 means no limit).",
    )
    parser.add_argument(
        "--history-source",
        choices=["export", "scan"],
        default="export",
        help=(
            "History fetch backend. "
            "'export' uses run.download_history_exports() parquet files (full-fidelity, no sampling). "
            "'scan' uses run.scan_history()."
        ),
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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # JSON does not represent NaN/Inf consistently; keep these stable as strings.
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        values = [_json_safe(v) for v in value]
        return sorted(values, key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), default=str))
    return str(value)


def _encode_complex(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return str(value) if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, (Mapping, list, tuple, set)):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _flatten_mapping(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, value in values.items():
        key = str(raw_key)
        if not key:
            continue
        col = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            if value:
                out.update(_flatten_mapping(value, col))
            else:
                out[col] = None
            continue
        out[col] = _encode_complex(value)
    return out


def _sanitize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, value in row.items():
        key = str(raw_key)
        if not key:
            continue
        if isinstance(value, Mapping):
            out.update(_flatten_mapping(value, key))
        else:
            out[key] = _encode_complex(value)
    return out


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    # strict=False allows mixed per-row types to coerce safely instead of hard-failing.
    try:
        return pl.from_dicts(rows, infer_schema_length=None, strict=False)
    except TypeError:
        # Backward compatibility with older Polars that may not expose strict.
        return pl.from_dicts(rows, infer_schema_length=None)


def _to_non_large_arrow_type(dtype: pa.DataType) -> pa.DataType:
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_large_binary(dtype):
        return pa.binary()
    if pa.types.is_large_list(dtype):
        value_type = _to_non_large_arrow_type(dtype.value_type)
        return pa.list_(value_type)
    if pa.types.is_struct(dtype):
        fields = [
            pa.field(
                field.name,
                _to_non_large_arrow_type(field.type),
                nullable=field.nullable,
                metadata=field.metadata,
            )
            for field in dtype
        ]
        return pa.struct(fields)
    return dtype


def _to_non_large_arrow_table(df: pl.DataFrame) -> pa.Table:
    table = df.to_arrow()
    schema = table.schema
    new_fields = []
    changed = False
    for field in schema:
        new_type = _to_non_large_arrow_type(field.type)
        if new_type != field.type:
            changed = True
            new_fields.append(pa.field(field.name, new_type, nullable=field.nullable, metadata=field.metadata))
        else:
            new_fields.append(field)
    if not changed:
        return table
    return table.cast(pa.schema(new_fields))


def _write_df(df: pl.DataFrame, output_path: Path, output_format: str) -> None:
    if output_format == "csv":
        df.write_csv(output_path)
        return
    if output_format == "parquet":
        # Normalize Arrow "large_*" logical types for downstream consumers that require standard types.
        pq.write_table(_to_non_large_arrow_table(df), output_path)
        return
    raise ValueError(f"Unsupported tabular output format: {output_format}")


def _run_to_row(run: wandb.apis.public.Run, sweep_id_override: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "sweep_id": sweep_id_override or (run.sweep.id if run.sweep else None),
    }
    row.update(_flatten_mapping({k: v for k, v in run.config.items() if not k.startswith("_")}, "config"))
    summary = getattr(run.summary, "_json_dict", dict(run.summary))
    row.update(_flatten_mapping({k: v for k, v in summary.items() if not k.startswith("_")}))
    return _sanitize_row(row)


def _collect_history_rows(
    run: wandb.apis.public.Run,
    sweep_id_override: str | None,
    history_keys: list[str] | None,
    history_max_rows: int | None,
    include_config: bool = True,
    history_source: str = "export",
) -> list[dict[str, Any]]:
    """Collect history rows for a run, including metadata and optionally config.

    Important: scan_history(keys=...) can drop rows when not all keys co-occur at a step.
    To avoid sparse-step artifacts, we always scan full history and apply key filtering locally.
    """
    rows = []
    base_row: dict[str, Any] = {
        "run_id": run.id,
        "run_name": run.name,
        "sweep_id": sweep_id_override or (run.sweep.id if run.sweep else None),
    }
    if include_config:
        base_row.update(_flatten_mapping({k: v for k, v in run.config.items() if not k.startswith("_")}, "config"))

    requested_keys = set(history_keys or [])

    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        return True

    def _process_history_rows(history_iter: Iterable[Mapping[str, Any]]) -> None:
        count = 0
        for row in history_iter:
            history_part = _sanitize_row(row)
            if requested_keys:
                # Keep the row only if any requested key has a real value at this step.
                if not any((key in history_part and _has_value(history_part[key])) for key in requested_keys):
                    continue
                history_part = {key: value for key, value in history_part.items() if key in requested_keys}

            row_out = base_row.copy()
            row_out.update(history_part)
            rows.append(_sanitize_row(row_out))
            count += 1
            if history_max_rows is not None and count >= history_max_rows:
                break

    if history_source == "export":
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                run.download_history_exports(download_dir=tmp_dir)
                parquet_files = sorted(Path(tmp_dir).rglob("*.parquet"))
                if parquet_files:
                    history_df = pl.concat([pl.read_parquet(pq) for pq in parquet_files], how="diagonal_relaxed")
                    _process_history_rows(history_df.to_dicts())
                    return rows
                print(f"  WARNING: no history export parquet files for run {run.id}; falling back to scan_history()")
        except Exception as exc:
            print(f"  WARNING: history export failed for run {run.id}: {exc}; falling back to scan_history()")

    _process_history_rows(run.scan_history())
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

    df = _rows_to_df(history_rows)
    _write_df(df, output_path, output_format)


def _fetch_project_runs(
    api: wandb.Api,
    project_path: str,
    max_runs: int,
    states: set[str] | None,
    collect_history: bool = False,
    history_keys: list[str] | None = None,
    history_max_rows: int | None = None,
    history_include_config: bool = True,
    history_source: str = "export",
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
                    history_source=history_source,
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
    history_source: str = "export",
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
                        history_source=history_source,
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

    df = _rows_to_df(rows)
    _write_df(df, output_path, output_format)


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
    history_source = args.history_source
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
            history_source=history_source,
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
            history_source=history_source,
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
