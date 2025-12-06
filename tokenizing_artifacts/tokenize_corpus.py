#!/usr/bin/env python3
"""Tokenize a corpus and save as memmap-compatible numpy array."""

import numpy as np
from cs336_basics.bpe import load_bpe, Tokenizer

INPUT_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
TOKENIZER_PATH = "./tokenizers/tinystories/"
OUTPUT_PATH = "./tokenized/tinystories_train.npy"

# Load tokenizer
vocab, merges = load_bpe(TOKENIZER_PATH)
tokenizer = Tokenizer(vocab, merges)

# Read and encode
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)

# Save as contiguous array (memmap-compatible)
arr = np.array(tokens, dtype=np.uint16)
np.save(OUTPUT_PATH, arr)

print(f"Saved {len(tokens):,} tokens to {OUTPUT_PATH}")

