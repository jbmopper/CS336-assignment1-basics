"""Byte Pair Encoding (BPE) tokenizer training and inference."""

import os
from typing import BinaryIO
import multiprocessing
import regex as re
from collections import Counter
from collections.abc import Iterable, Iterator
import json
import pickle
from dataclasses import dataclass, field

__all__ = ['train_bpe', 'Tokenizer']

# GPT-2 style pretokenization pattern
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the input corpus.
    
    Args:
        input_path: Path to the training corpus
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to add to vocabulary
    
    Returns:
        Tuple of (vocabulary dict, list of merges)
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    pretoken_counts: Counter[str] = Counter()
    paircount: Counter[tuple[bytes, bytes]] = Counter()

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, 8, b"<|endoftext|>")

    chunks = zip(boundaries[:-1], boundaries[1:])
    with multiprocessing.Pool(processes=8) as pool:
        chunked_counters = pool.map(
            _worker, [(input_path, s, e, special_tokens, GPT2_PAT) for s, e in chunks]
        )
        for cc in chunked_counters:
            pretoken_counts.update(cc)

    tokenized = {
        pretoken: [bytes([b]) for b in pretoken.encode("utf-8")]
        for pretoken in pretoken_counts.keys()
    }

    # First pass - count initial pairs
    for pretoken, b in tokenized.items():
        for i in range(len(b) - 1):
            paircount[(b[i], b[i + 1])] += pretoken_counts[pretoken]

    while len(vocab) < vocab_size:
        merge = max(paircount.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        _update_tokens(merge, tokenized, paircount, pretoken_counts)

    return vocab, merges


def _build_pretokenizer(special_tokens: list[str]) -> str:
    """Build regex pattern that handles special tokens."""
    escaped = [re.escape(token) for token in special_tokens]
    special_pattern = "|".join(escaped)
    if special_pattern:
        return f"{special_pattern}|{GPT2_PAT}"
    return GPT2_PAT


def _worker(args):
    """Worker function for parallel pretokenization."""
    path, start, end, special_tokens, pat = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return _pretokenize(chunk, special_tokens, pat)


def _pretokenize(chunk: str, special_tokens: list[str], pretokenizer: str) -> Counter[str]:
    """Pretokenize a chunk of text."""
    pretoken_counts = Counter()
    specials_removed = re.split("|".join(re.escape(st) for st in special_tokens), chunk)
    for piece in specials_removed:
        pretokens = re.finditer(pretokenizer, piece)
        for pretoken in pretokens:
            pretoken_counts[pretoken.group(0)] += 1
    return pretoken_counts


def _update_tokens(
    merge: tuple[bytes, bytes],
    tokenized: dict[str, list[bytes]],
    paircount: Counter,
    pretoken_counts: Counter
) -> None:
    """Update token lists and pair counts after a merge."""
    merged = merge[0] + merge[1]
    paircount[merge] = 0
    
    for pretoken, b in tokenized.items():
        count = pretoken_counts[pretoken]
        
        # Scan and identify merges
        merge_idx = []
        updated_tokens = []
        i = 0
        while i < len(b):
            if i < len(b) - 1 and b[i] == merge[0] and b[i + 1] == merge[1]:
                updated_tokens.append(merged)
                merge_idx.append(i)
                merge_idx.append(i + 1)
                i += 2
            else:
                updated_tokens.append(b[i])
                i += 1

        merge_starts = []
        merge_ends = []
        internal_merges = []
        
        for i in range(len(merge_idx)):
            if merge_idx[i] - 1 not in merge_idx:
                merge_starts.append(merge_idx[i])
            elif merge_idx[i] + 1 not in merge_idx:
                merge_ends.append(merge_idx[i])
            else:
                internal_merges.append(merge_idx[i])

        for i in merge_starts:
            if i > 0:
                paircount[(b[i - 1], b[i])] -= count
                paircount[(b[i - 1], merged)] += count

        for i in merge_ends:
            if i < len(b) - 1:
                paircount[(b[i], b[i + 1])] -= count
                paircount[(merged, b[i + 1])] += count

        paircount[(merge[1], merge[0])] -= count * (len(internal_merges) // 2)
        paircount[(merged, merged)] += count * (len(internal_merges) // 2)

        tokenized[pretoken] = updated_tokens

    del paircount[merge]


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find chunk boundaries for parallel processing.
    
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


@dataclass
class TrieNode:
    """Trie node for efficient token lookup."""
    children: dict = field(default_factory=dict)
    token_id: int | None = None


class Tokenizer:
    """BPE Tokenizer for encoding and decoding text."""
    
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.trie_root = self._build_trie()
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_prios = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Load tokenizer from vocab and merges files."""
        with open(vocab_filepath, "rb") as f:
            vocab_raw = json.load(f)

        vocab = {int(k): v.encode("utf-8") for k, v in vocab_raw.items()}
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_token not in vocab.values():
                    vocab[len(vocab)] = special_token.encode("utf-8")

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _build_trie(self) -> TrieNode:
        """Build trie from vocabulary for efficient lookup - wrong approach."""
        root = TrieNode()
        for token_id, bytes_ in self.vocab.items():
            node = root
            for byte in bytes_:
                k = bytes([byte])
                if k not in node.children:
                    node.children[k] = TrieNode()
                node = node.children[k]
            node.token_id = token_id
        return root

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        encoded: list[int] = []
        pretokenizer = GPT2_PAT
        
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(st) for st in sorted_special) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        

        for part in parts:
            if part in self.special_tokens:
                encoded.append(self.byte_to_id[part.encode("utf-8")])
            elif part:
                pretokens = re.finditer(pretokenizer, part)
                for pretoken in pretokens:
                    tokens = [bytes([b]) for b in pretoken.group(0).encode("utf-8", errors="ignore")]
                    # rewrite rewrite rewrite
                    # 
                    merge_idx = None # will be the index
                    merges_exhausted = False
                    while merges_exhausted is not True:
                        highest_prio = float('inf')
                        for i in range(len(tokens) - 1):
                            if (tokens[i], tokens[i+1]) in self.merge_prios:
                                if (p:= self.merge_prios[(tokens[i], tokens[i+1])]) < highest_prio:
                            # if p := self.merge_prios.get((tokens[i], tokens[i+1]), float('inf')) < highest_prio:
                                # highest prio = lowest p!
                                    merge_idx = i
                                    highest_prio = p

                        if merge_idx is not None:
                            tokens[merge_idx] = tokens[merge_idx] + tokens[merge_idx+1]
                            tokens.pop(merge_idx + 1)
                            merge_idx = None
                        else:
                            merges_exhausted = True

                    
                    for token in tokens:
                        encoded.append(self.byte_to_id[token])
                            


                    


                    # for merge0, merge1 in self.merges:
                    #     i = 0
                    #     while i < len(tokens) - 1:
                    #         if tokens[i] == merge0 and tokens[i + 1] == merge1:
                    #             tokens[i] = merge0 + merge1
                    #             tokens.pop(i + 1)
                    #         else:
                    #             i += 1
                    # for token in tokens:
                    #     encoded.append(self.byte_to_id[token])

        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of text chunks, yielding token IDs."""
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        result = b""
        for id_ in ids:
            result += self.vocab[id_]
        return result.decode("utf-8", errors="replace")
