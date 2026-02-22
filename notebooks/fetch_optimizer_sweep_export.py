#!/usr/bin/env python3
"""
Fetch a focused optimizer sweep export from W&B.

This wrapper uses notebooks/wandb_fetch_runs.py to export:
1) run-level data (configs + summaries)
2) history-level data (selected keys, all available steps per run)

Output filenames are timestamped and made non-colliding in notebooks/benchmark_results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_PROJECT = "jbmopper-0/cs336-optimizer-sweep-high"
DEFAULT_SWEEP_ID = "l28yv8f7"
DEFAULT_PREFIX = "optimizer_sweep"
DEFAULT_HISTORY_KEYS = [
    "Sweep Metric M",
    "Loss",
    "LR",
    "Eval Loss",
    "Eval Best loss",
    "Grad/Norm (clipped)",
    "Grad/Norm (unclipped)",
    "Time/Total step",
    "Throughput/Tokens per sec",
    "_step",
]


def _non_colliding_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def _parse_history_keys(value: str) -> list[str]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise SystemExit("History keys cannot be empty.")
    return keys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch optimizer sweep parquets.")
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project path in the form 'entity/project'.",
    )
    parser.add_argument(
        "--sweep-id",
        default=DEFAULT_SWEEP_ID,
        help="W&B sweep ID to export.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("notebooks/benchmark_results"),
        help="Directory where parquet outputs will be written.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Filename prefix for output artifacts.",
    )
    parser.add_argument(
        "--history-keys",
        default=",".join(DEFAULT_HISTORY_KEYS),
        help="Comma-separated history keys to export.",
    )
    parser.add_argument(
        "--history-max-rows",
        type=int,
        default=0,
        help="Max history rows per run (0 means all available rows).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Max runs to fetch from the sweep (0 means all runs).",
    )
    parser.add_argument(
        "--states",
        default="",
        help="Optional comma-separated run states filter (empty means all states).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and planned output paths without running.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    history_keys = _parse_history_keys(args.history_keys)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    fetch_script = script_dir / "wandb_fetch_runs.py"
    if not fetch_script.exists():
        raise SystemExit(f"Missing fetch script: {fetch_script}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.prefix}_{args.sweep_id}_{timestamp}"
    main_path = _non_colliding_path(args.out_dir / f"{base_name}_main.parquet")
    history_path = _non_colliding_path(args.out_dir / f"{base_name}_history.parquet")
    manifest_path = _non_colliding_path(args.out_dir / f"{base_name}_manifest.json")

    cmd: list[str] = [
        sys.executable,
        str(fetch_script),
        "--project",
        args.project,
        "--sweep-ids",
        args.sweep_id,
        "--include-history",
        "--history-keys",
        ",".join(history_keys),
        "--history-max-rows",
        str(args.history_max_rows),
        "--history-output",
        str(history_path),
        "--output",
        str(main_path),
    ]
    if args.max_runs > 0:
        cmd.extend(["--max-runs", str(args.max_runs)])
    if args.states.strip():
        cmd.extend(["--states", args.states.strip()])

    print("Planned outputs:")
    print(f"  main:    {main_path}")
    print(f"  history: {history_path}")
    print(f"  manifest:{manifest_path}")
    print("")
    print("Running command:")
    print("  " + " ".join(cmd))

    if args.dry_run:
        return 0

    subprocess.run(cmd, cwd=repo_root, check=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "project": args.project,
        "sweep_id": args.sweep_id,
        "history_keys": history_keys,
        "history_max_rows": args.history_max_rows,
        "max_runs": args.max_runs,
        "states": args.states,
        "outputs": {
            "main": str(main_path),
            "history": str(history_path),
        },
        "command": cmd,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("")
    print("Done.")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
