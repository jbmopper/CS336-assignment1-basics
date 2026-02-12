#!/usr/bin/env python3
"""
Run wandb_fetch_runs.py for each project in notebooks/export/projects.yaml.

For each key/value in projects.yaml, runs:
  uv run notebooks/wandb_fetch_runs.py
    --project <value>
    --all-runs
    --include-history
    --history-output notebooks/export/<key>_history.parquet
    --output notebooks/export/<key>_main.parquet

Usage (from repo root):
  uv run notebooks/fetch_export_projects.py
  uv run notebooks/fetch_export_projects.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def load_projects(config_path: Path) -> dict[str, str]:
    """Load key: project mapping from a simple 'key: value' per line file."""
    projects = {}
    with config_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value:
                projects[key] = value
    return projects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("notebooks/export/projects.yaml"),
        help="Path to projects.yaml (default: notebooks/export/projects.yaml).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("notebooks/export"),
        help="Directory for output files (default: notebooks/export).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    projects = load_projects(args.config)
    if not projects:
        print("No projects found in config.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    fetch_script = script_dir / "wandb_fetch_runs.py"

    for key, project in projects.items():
        main_out = args.out_dir / f"{key}_main.parquet"
        history_out = args.out_dir / f"{key}_history.parquet"
        cmd = [
            "uv",
            "run",
            str(fetch_script),
            "--project",
            project,
            "--all-runs",
            "--include-history",
            "--history-output",
            str(history_out),
            "--output",
            str(main_out),
        ]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"Fetching {key} ({project}) -> {main_out.name}, {history_out.name} ...")
        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"Failed: {key}", file=sys.stderr)
            return result.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
