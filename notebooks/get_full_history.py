#!/usr/bin/env python3
"""Download complete W&B history exports for projects listed in projects_to_download.yaml.

Uses run.download_history_exports() to get the full, un-sampled history
as Parquet files directly from W&B's export backend.

Usage:
    uv run notebooks/get_full_history.py
    uv run notebooks/get_full_history.py --only gpu_comp_history
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_FILE = SCRIPT_DIR / "projects_to_download.yaml"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"


def _downcast_type(t: pa.DataType) -> pa.DataType:
    ts = str(t)
    if ts == "large_string":
        return pa.string()
    if ts == "large_binary":
        return pa.binary()
    if pa.types.is_large_list(t):
        return pa.list_(_downcast_type(t.value_type))
    if pa.types.is_struct(t):
        return pa.struct([
            pa.field(f.name, _downcast_type(f.type), nullable=f.nullable)
            for f in t
        ])
    return t


def downcast_large_types(table: pa.Table) -> pa.Table:
    new_schema = pa.schema([
        pa.field(f.name, _downcast_type(f.type), nullable=f.nullable)
        for f in table.schema
    ])
    if new_schema == table.schema:
        return table
    return table.cast(new_schema)


def fetch_project_history(api: wandb.Api, project_path: str) -> pl.DataFrame:
    """Download full history exports for every run in a project."""
    runs = api.runs(project_path)
    all_dfs: list[pl.DataFrame] = []

    for run_idx, run in enumerate(runs):
        t0 = time.monotonic()

        with tempfile.TemporaryDirectory() as tmp_dir:
            run.download_history_exports(download_dir=tmp_dir)

            parquet_files = list(Path(tmp_dir).rglob("*.parquet"))
            if not parquet_files:
                print(f"  [{run_idx + 1}] {run.name} ({run.id}): no export files")
                continue

            run_df = pl.concat([pl.read_parquet(f) for f in parquet_files])

        run_df = run_df.with_columns([
            pl.lit(run.id).alias("run_id"),
            pl.lit(run.name).alias("run_name"),
            pl.lit(run.state).alias("state"),
            pl.lit(run.sweep.id if run.sweep else None).alias("sweep_id"),
        ])

        all_dfs.append(run_df)
        elapsed = time.monotonic() - t0
        print(f"  [{run_idx + 1}] {run.name} ({run.id}, {run.state}): {len(run_df):,} rows in {elapsed:.1f}s")

    if not all_dfs:
        print(f"  WARNING: no history found for {project_path}")
        return pl.DataFrame()

    return pl.concat(all_dfs, how="diagonal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Process only this key from the YAML file.",
    )
    args = parser.parse_args()

    with open(PROJECTS_FILE) as f:
        projects: dict[str, str] = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=120)

    for key, project_path in projects.items():
        if args.only and key != args.only:
            continue

        output_file = OUTPUT_DIR / f"{key}.parquet"
        print(f"\n{'=' * 60}")
        print(f"Project: {project_path}")
        print(f"Output:  {output_file}")
        print(f"{'=' * 60}")

        t0 = time.monotonic()
        df = fetch_project_history(api, project_path)
        elapsed = time.monotonic() - t0

        if df.is_empty():
            print(f"  Skipping write (empty dataframe).")
            continue

        table = downcast_large_types(df.to_arrow())
        pq.write_table(table, output_file)
        print(f"  Wrote {len(df):,} rows, {df.width} columns to {output_file}")
        print(f"  Total time: {elapsed:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
