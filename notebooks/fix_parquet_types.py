#!/usr/bin/env python3
"""Rewrite parquet files in-place to replace large_* Arrow types with standard ones.

Fixes parquet-wasm compatibility by converting large_string -> string,
large_list -> list, etc.

Usage:
    uv run notebooks/fix_parquet_types.py
    uv run notebooks/fix_parquet_types.py --dry-run   # just report, don't rewrite
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmark_results"


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't rewrite")
    args = parser.parse_args()

    files = sorted(BENCHMARK_DIR.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {BENCHMARK_DIR}")
        return

    for f in files:
        schema = pq.read_schema(f)
        large_cols = [field for field in schema if "large" in str(field.type).lower()]

        if not large_cols:
            print(f"  {f.name}: already clean")
            continue

        print(f"  {f.name}: {len(large_cols)} large types")
        if args.dry_run:
            for field in large_cols:
                print(f"    {field.name}: {field.type}")
            continue

        table = pq.read_table(f)
        table = downcast_large_types(table)
        pq.write_table(table, f)
        print(f"    -> rewritten")

    print("\nDone.")


if __name__ == "__main__":
    main()
