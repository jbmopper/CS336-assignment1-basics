#!/usr/bin/env python3
"""Batch export all best checkpoints in a run directory to ONNX."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export every */best.pt checkpoint in a run directory to ONNX."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory with model subfolders.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write ONNX files into.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size.")
    parser.add_argument("--seq-len", type=int, default=64, help="Dummy export sequence length.")
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="Export fixed-shape graph instead of dynamic axes.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ONNXRuntime parity validation (off by default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ONNX outputs if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned exports without running them.",
    )
    return parser.parse_args()


def canonical_name(run_subdir_name: str) -> str:
    # Remove trailing timestamp suffix like _20260223_0332.
    return re.sub(r"_\d{8}_\d{4,6}$", "", run_subdir_name)


def find_best_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("*/best.pt") if p.is_file())


def main() -> int:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    best_ckpts = find_best_checkpoints(run_dir)
    if not best_ckpts:
        raise FileNotFoundError(f"No best.pt files found under: {run_dir}")

    export_plan: list[tuple[Path, Path, Path]] = []
    seen_outputs: set[Path] = set()
    for ckpt in best_ckpts:
        exp_dir = ckpt.parent
        model_name = canonical_name(exp_dir.name)
        prefill_out_path = out_dir / f"{model_name}_prefill_best.onnx"
        decode_out_path = out_dir / f"{model_name}_decode_best.onnx"
        for out_path in (prefill_out_path, decode_out_path):
            if out_path in seen_outputs:
                raise ValueError(
                    f"Output filename collision for {out_path.name}. "
                    "Use unique run subdirectory names."
                )
            seen_outputs.add(out_path)
        export_plan.append((ckpt, prefill_out_path, decode_out_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(export_plan)} best checkpoints under {run_dir}")
    for ckpt, prefill_out_path, decode_out_path in export_plan:
        if (prefill_out_path.exists() or decode_out_path.exists()) and not args.overwrite:
            print(f"SKIP (exists): {prefill_out_path} or {decode_out_path}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "cs336_basics.export_onnx",
            "--checkpoint",
            str(ckpt),
            "--prefill_out",
            str(prefill_out_path),
            "--decode_out",
            str(decode_out_path),
            "--opset",
            str(args.opset),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
        ]
        if args.static_shapes:
            cmd.append("--static-shapes")
        if not args.validate:
            cmd.append("--skip-validate")

        print(f"EXPORT: {ckpt} -> {prefill_out_path}, {decode_out_path}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
