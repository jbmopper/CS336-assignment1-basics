"""Minimal BPE tokenizer runtime for Lambda inference."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Iterator

import regex as re

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def load_bpe(tokenizer_dir: str | Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    root = Path(tokenizer_dir)
    vocab_path = root / "vocab.json"
    merges_path = root / "merges.pkl"

    with vocab_path.open("r", encoding="latin-1") as f:
        vocab_str = json.load(f)
    vocab = {int(k): v.encode("latin-1") for k, v in vocab_str.items()}

    with merges_path.open("rb") as f:
        merges = pickle.load(f)
    return vocab, merges

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merge_prios = {merge: i for i, merge in enumerate(merges)}

        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = "(" + "|".join(re.escape(st) for st in sorted_special) + ")"
        else:
            self._special_pattern = None

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        if self._special_pattern:
            parts = re.split(self._special_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                encoded.append(self.byte_to_id[part.encode("utf-8")])
                continue

            for pretoken in re.finditer(GPT2_PAT, part):
                tokens = [bytes([b]) for b in pretoken.group(0).encode("utf-8", errors="ignore")]
                while True:
                    merge_idx = None
                    best_prio = float("inf")
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        if pair in self.merge_prios and self.merge_prios[pair] < best_prio:
                            best_prio = self.merge_prios[pair]
                            merge_idx = i
                    if merge_idx is None:
                        break
                    tokens[merge_idx] = tokens[merge_idx] + tokens[merge_idx + 1]
                    tokens.pop(merge_idx + 1)

                for tok in tokens:
                    encoded.append(self.byte_to_id[tok])
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token in self.encode(chunk):
                yield token

    def decode(self, ids: list[int]) -> str:
        out = b"".join(self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")
