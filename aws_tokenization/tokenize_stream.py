#!/usr/bin/env python3
"""Stream-tokenize a large corpus into a .npy array via a temp .bin."""

from __future__ import annotations

import argparse
import os
from typing import Iterator

import numpy as np

from cs336_basics.bpe import load_bpe, Tokenizer


DEFAULT_SPECIAL = "<|endoftext|>"


def iter_docs(path: str, chunk_bytes: int, special_token: str) -> Iterator[bytes]:
    """Yield document-sized byte chunks split by special token."""
    special_bytes = special_token.encode("utf-8")
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


def iter_text_docs(path: str, chunk_bytes: int, special_token: str) -> Iterator[str]:
    """Decode byte chunks into text safely."""
    for doc in iter_docs(path, chunk_bytes, special_token):
        yield doc.decode("utf-8", errors="ignore")


def write_tokens_to_bin(
    tokenizer: Tokenizer,
    input_path: str,
    bin_path: str,
    chunk_bytes: int,
    special_token: str,
    flush_tokens: int,
    max_docs: int | None,
) -> int:
    """Tokenize input and write uint16 tokens to a binary file."""
    token_count = 0
    buffer: list[int] = []
    docs_seen = 0

    with open(bin_path, "wb") as out:
        for doc in iter_text_docs(input_path, chunk_bytes, special_token):
            buffer.extend(tokenizer.encode(doc))
            if len(buffer) >= flush_tokens:
                arr = np.array(buffer, dtype=np.uint16)
                out.write(arr.tobytes(order="C"))
                token_count += len(arr)
                buffer.clear()
            docs_seen += 1
            if max_docs is not None and docs_seen >= max_docs:
                break

        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            out.write(arr.tobytes(order="C"))
            token_count += len(arr)

    return token_count


def bin_to_npy(bin_path: str, npy_path: str, block_tokens: int) -> None:
    """Convert a uint16 .bin file to .npy using memmaps."""
    file_size = os.path.getsize(bin_path)
    token_count = file_size // 2

    src = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(token_count,))
    dst = np.lib.format.open_memmap(
        npy_path, dtype=np.uint16, mode="w+", shape=(token_count,)
    )

    for start in range(0, token_count, block_tokens):
        end = min(start + block_tokens, token_count)
        dst[start:end] = src[start:end]
        dst.flush()

    del dst
    del src


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-tokenize a corpus.")
    parser.add_argument("input_path", help="Path to corpus text file.")
    parser.add_argument(
        "--tokenizer-dir",
        default="tokenizers/owt",
        help="Directory containing vocab.json and merges.pkl.",
    )
    parser.add_argument(
        "--output",
        default="tokenized/owt_train.npy",
        help="Output .npy path (default: tokenized/owt_train.npy).",
    )
    parser.add_argument(
        "--bin-path",
        default=None,
        help="Optional path for temporary .bin output.",
    )
    parser.add_argument(
        "--special-token",
        default=DEFAULT_SPECIAL,
        help="Special token used as document boundary.",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Bytes to read per chunk (default: 64MB).",
    )
    parser.add_argument(
        "--flush-tokens",
        type=int,
        default=5_000_000,
        help="Token buffer size before flushing to disk.",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=10_000_000,
        help="Tokens per block when copying .bin to .npy.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on documents processed (debug only).",
    )
    parser.add_argument(
        "--keep-bin",
        action="store_true",
        help="Keep the intermediate .bin file.",
    )
    args = parser.parse_args()

    vocab, merges = load_bpe(args.tokenizer_dir)
    tokenizer = Tokenizer(vocab, merges, special_tokens=[args.special_token])

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    bin_path = args.bin_path or f"{args.output}.bin"

    print("Tokenizing...")
    token_count = write_tokens_to_bin(
        tokenizer,
        args.input_path,
        bin_path,
        args.chunk_bytes,
        args.special_token,
        args.flush_tokens,
        args.max_docs,
    )
    print(f"Wrote {token_count:,} tokens to {bin_path}")

    print("Converting to .npy...")
    bin_to_npy(bin_path, args.output, args.block_tokens)
    print(f"Saved {args.output}")

    if not args.keep_bin:
        os.remove(bin_path)


if __name__ == "__main__":
    main()
