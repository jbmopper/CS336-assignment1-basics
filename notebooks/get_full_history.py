#!/usr/bin/env python3
"""Download complete W&B history for projects listed in projects_to_download.yaml.

Uses run.scan_history() WITHOUT passing keys=None to avoid the sampled-paging
bug described in W&B support threads.  Each project's full history (all runs,
all steps) is saved as a single Parquet file in notebooks/benchmark_results/.

Usage:
    uv run notebooks/get_full_history.py
    uv run notebooks/get_full_history.py --page-size 500   # lower page size if timeouts
    uv run notebooks/get_full_history.py --only gpu_comp_history  # single project
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import wandb
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECTS_FILE = SCRIPT_DIR / "projects_to_download.yaml"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"


def _encode_value(value: Any) -> Any:
    """Make a value safe for Polars / Parquet serialization."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (Mapping, list, tuple, set)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def _flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dicts with dot-separated keys."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = _encode_value(value)
    return out


def fetch_project_history(
    api: wandb.Api,
    project_path: str,
    page_size: int,
) -> pl.DataFrame:
    """Fetch all history rows for every run in a project.

    Calls scan_history() with no keys argument so W&B returns un-sampled,
    complete rows via its streaming/paging path.
    """
    runs = api.runs(project_path)
    all_rows: list[dict[str, Any]] = []

    for run_idx, run in enumerate(runs):
        run_id = run.id
        run_name = run.name
        state = run.state
        sweep_id = run.sweep.id if run.sweep else None

        config_flat = _flatten(
            {k: v for k, v in run.config.items() if not k.startswith("_")},
            prefix="config",
        )

        run_meta = {
            "run_id": run_id,
            "run_name": run_name,
            "state": state,
            "sweep_id": sweep_id,
            **config_flat,
        }

        row_count = 0
        t0 = time.monotonic()
        for row in run.scan_history(page_size=page_size):
            merged = {**run_meta}
            for k, v in row.items():
                if k.startswith("_") and k != "_step":
                    continue
                merged[k] = _encode_value(v)
            all_rows.append(merged)
            row_count += 1

        elapsed = time.monotonic() - t0
        print(
            f"  [{run_idx + 1}] {run_name} ({run_id}, {state}): "
            f"{row_count:,} rows in {elapsed:.1f}s"
        )

    if not all_rows:
        print(f"  WARNING: no history rows found for {project_path}")
        return pl.DataFrame()

    try:
        return pl.from_dicts(all_rows, infer_schema_length=None, strict=False)
    except TypeError:
        return pl.from_dicts(all_rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Rows per page for scan_history (lower if you hit timeouts).",
    )
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
        df = fetch_project_history(api, project_path, page_size=args.page_size)
        elapsed = time.monotonic() - t0

        if df.is_empty():
            print(f"  Skipping write (empty dataframe).")
            continue

        df.write_parquet(output_file)
        print(f"  Wrote {len(df):,} rows, {df.width} columns to {output_file}")
        print(f"  Total time: {elapsed:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
