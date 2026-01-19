#!/usr/bin/env python3
"""Sample pretoken statistics from a large corpus safely."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Iterator

from cs336_basics.bpe import GPT2_PAT, _pretokenize


SPECIAL_TOKEN = "<|endoftext|>"


def iter_docs(path: str, chunk_bytes: int) -> Iterator[bytes]:
    """Yield document-sized byte chunks split by SPECIAL_TOKEN."""
    special_bytes = SPECIAL_TOKEN.encode("utf-8")
    buffer = b""

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(special_bytes)
            for part in parts[:-1]:
                yield part + special_bytes
            buffer = parts[-1]

        if buffer:
            yield buffer


def sample_pretokens(
    path: str,
    max_docs: int | None,
    chunk_bytes: int,
) -> tuple[Counter[str], int]:
    """Return pretoken counts and the number of docs scanned."""
    counts: Counter[str] = Counter()
    docs_seen = 0

    for raw_doc in iter_docs(path, chunk_bytes):
        text = raw_doc.decode("utf-8", errors="ignore")
        counts.update(_pretokenize(text, [SPECIAL_TOKEN], GPT2_PAT))
        docs_seen += 1
        if max_docs is not None and docs_seen >= max_docs:
            break

    return counts, docs_seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample pretoken statistics.")
    parser.add_argument("path", help="Path to corpus text file.")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum documents to sample (default: 10k).",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Bytes to read per chunk (default: 64MB).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top pretokens to display (default: 20).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        hint_path = args.path.replace("owd_valid.txt", "owt_valid.txt")
        if hint_path != args.path and os.path.exists(hint_path):
            print(f"File not found: {args.path}")
            print(f"Did you mean: {hint_path}")
            print("Run again with the corrected path.")
            sys.exit(2)
        raise FileNotFoundError(f"No such file: {args.path}")

    counts, docs_seen = sample_pretokens(args.path, args.max_docs, args.chunk_bytes)

    if args.max_docs is not None and docs_seen < args.max_docs:
        print(f"Docs sampled: {docs_seen} (EOF reached)")
    else:
        print(f"Docs sampled: {docs_seen}")
    print(f"Unique pretokens: {len(counts):,}")
    print(f"Top {args.top_k} pretokens:")
    for token, count in counts.most_common(args.top_k):
        print(f"{count:>10,}  {token!r}")


if __name__ == "__main__":
    main()
