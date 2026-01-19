#!/usr/bin/env python3
"""Train and save a BPE tokenizer for a corpus."""

from __future__ import annotations

import argparse
import time

from cs336_basics.bpe import save_bpe, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("input_path", help="Path to training corpus text file.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size (default: 32000).",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizers/owt",
        help="Directory to save vocab/merges (default: tokenizers/owt).",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token (repeatable). Default: <|endoftext|>.",
    )
    args = parser.parse_args()

    start = time.time()
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
    )
    elapsed = time.time() - start
    print(f"Training took {elapsed:.2f}s")

    save_bpe(args.output_dir, vocab, merges)
    print(f"Saved tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
