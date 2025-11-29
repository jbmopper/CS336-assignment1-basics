import os
from typing import Any, BinaryIO
import multiprocessing
import regex as re
from collections import Counter
from collections.abc import Iterable, Iterator
import json
import pickle

from dataclasses import dataclass, field

from torch import mul

__all__ =   ['train_bpe', 'Tokenizer']
# __all__ =   ['train_bpe']

def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # PAT = build_pretokenizer(special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {i: bytes([i]) for i in range(256)} # also need special tokens?
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = [] # merges.append((b"urg", b"bla"))
    pretoken_counts: Counter[str] = Counter()
    paircount: Counter[tuple[bytes, bytes]] = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 8, b"<|endoftext|>") # not special token agnostic...

    chunks = zip(boundaries[:-1], boundaries[1:])
    with multiprocessing.Pool(processes=8) as pool:
        chunked_counters = pool.map(
            _worker, [(input_path, s, e, special_tokens, PAT) for s, e in chunks]
            )
        
        for cc in chunked_counters:
            pretoken_counts.update(cc)
        
        # if __name__ == "__main__":
        #    with multiprocessing.Pool(...) as pool:
        #        chunked_counters = pool.map(lambda x: pretokenize(f, x, PAT), chunks)

        
    # del pretoken_counts['<|']
    # del pretoken_counts['|>']
    # del pretoken_counts['endoftext']
        
    tokenized = { # need to actually merge bytes
        pretoken: [bytes([b]) for b in pretoken.encode("utf-8")]
        for pretoken in pretoken_counts.keys()
    }

    # for special_token in special_tokens:
    #     tokenized[special_token] = [special_token.encode("utf-8")]
    #     pretoken_counts[special_token] = 0 # 

    # first pass    
    for pretoken, b in tokenized.items():
        for i in range(len(b)-1): # skips len 1 b's
            paircount[(b[i], b[i+1])] += 1 * pretoken_counts[pretoken]

    while len(vocab) < vocab_size:
        # need the byte gumming here
        merge = max(paircount.items(), key=lambda kv: (kv[1], kv[0]))[0] # bot says this does it all...
        merges.append(merge)
        vocab[len(vocab)] = merge[0] + merge[1]
        update_tokens_v2(merge, tokenized, paircount, pretoken_counts)

    return vocab, merges


def build_pretokenizer(special_tokens: list[str]) -> str: # using for encode
    # Escape regex special characters in tokens
    escaped = [re.escape(token) for token in special_tokens]
    special_pattern = "|".join(escaped)
    # # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
    base_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    if special_pattern:
         return f"{special_pattern}|{base_pat}"
    return base_pat

def _worker(args):
    path, start, end, special_tokens, PAT = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize(chunk, special_tokens, PAT)


def pretokenize(
    chunk: str,
    special_tokens: list[str],
    pretokenizer: str
) -> Counter[str]:
    pretoken_counts = Counter()
    specials_removed =  re.split("|".join(re.escape(st) for st in special_tokens), chunk)
    for piece in specials_removed:
        pretokens = re.finditer(pretokenizer, piece)
        for pretoken in pretokens:
            pretoken_counts[pretoken.group(0)] += 1

    return pretoken_counts




def update_tokens( # updates token lists and pair counts
    merge: tuple[bytes, bytes],
    tokenized: dict[str, list[bytes]],
    paircount: Counter,
    pretoken_counts: Counter
) -> None:
    merged = merge[0] + merge[1]
    paircount[merge] = 0
    for pretoken, b in tokenized.items():
        count = pretoken_counts[pretoken] 
        # decrement prior to re-counting paris
        for i in range(len(b) - 1):
            paircount[(b[i], b[i+1])] -= count

        updated_tokens = []
        i = 0
        while i < len(b):
            if i < len(b) - 1 and b[i] == merge[0] and b[i+1] == merge[1]:
                updated_tokens.append(merged)
                i += 2
            else:
                updated_tokens.append(b[i])
                i += 1
        tokenized[pretoken] = updated_tokens

        for i in range(len(updated_tokens) - 1):
            paircount[(updated_tokens[i], updated_tokens[i+1])] += count

    # remove the merged pair from paircount
    del(paircount[merge])
    return None


def update_tokens_v2( # updates token lists and pair counts
    merge: tuple[bytes, bytes],
    tokenized: dict[str, list[bytes]],
    paircount: Counter,
    pretoken_counts: Counter
) -> None:
    merged = merge[0] + merge[1]
    paircount[merge] = 0
    for pretoken, b in tokenized.items():
        count = pretoken_counts[pretoken] 
        
        # scan the pretoken and identify merges
        merge_idx = []
        updated_tokens = []
        i = 0
        while i < len(b):
            if i < len(b) - 1 and b[i] == merge[0] and b[i+1] == merge[1]:
                updated_tokens.append(merged)
                merge_idx.append(i)
                merge_idx.append(i+1)
                i += 2
            else:
                updated_tokens.append(b[i])
                i += 1

        # if b = [A, B, A, B], merging (A, B) → AB, results in [AB, AB]

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
            if i > 0: # deal with left edge
                paircount[(b[i-1], b[i])] -= count
                paircount[(b[i-1], merged)] += count

        for i in merge_ends:
            if i < len(b) - 1: 
                paircount[(b[i], b[i+1])] -= count
                paircount[(merged, b[i+1])] += count

        # for i in internal_merges:
        paircount[(merge[1], merge[0])] -= count * (len(internal_merges)//2)
        paircount[(merged, merged)] += count * (len(internal_merges)//2)
        


        tokenized[pretoken] = updated_tokens

    # remove the merged pair from paircount
    del(paircount[merge])
    return None



# from pretokenization_example.py
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


 
    # for get_tokenizer from adapters.py

@dataclass
class TrieNode():
    children: dict = field(default_factory=dict)
    token_id: int | None = None
    # def __init__(self):
    #     self.children: dict[bytes, "TrieNode"] = {}
    #     self.token_id: int | None = None


class Tokenizer(
#    vocab: dict[int, bytes],
#    merges: list[tuple[bytes, bytes]],
#    special_tokens: list[str] | None = None,
):
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens or []
        self.trie_root = self._build_trie()
        self._specials_dict = {}
        self.byte_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}
        return

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f: #vocab: dict[int, bytes]
           vocab_raw = json.load(f)

        vocab = {int(k): v.encode("utf-8") for k, v in vocab_raw.items()}   
        if special_tokens is not None:
            for special_token in special_tokens:
                if special_tokens not in vocab.values():
                    vocab[len(vocab)] = special_token.encode("utf-8")

        with open(merges_filepath, "rb") as f: #bot: just use pickle
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _build_trie(self) -> TrieNode:
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
        # so let's see... 
        # text = text.encode("utf-8") # makes a bytes object... errors?
        # want a list of pretokenized strings

        encoded: list[int] = []
        pretokenizer = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(st) for st in self.special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                encoded.append(self.byte_to_id[part.encode("utf-8")])
            elif part:
                pretokens = re.finditer(pretokenizer, part) # special tokens are elements in the iterator
                for pretoken in pretokens:
                    bytes_ = pretoken.group(0).encode("utf-8", errors="ignore") # strict?
                  #  if pretoken.group(0) in self.special_tokens:
                  #      encoded.append(self.byte_to_id[bytes_])
                  #  else:
                            # immutable list of byte integers
                            # ... get the longest match
                        
                    node = self.trie_root
                    # nodes = []
                    last: tuple[int, int] = None
                    i = 0
                    while i < len(bytes_):
                        key = bytes([bytes_[i]])
                        if key in node.children:
                            node = node.children[key]
                            if node.token_id is not None:
                                last = (i, node.token_id)
                            i += 1
                        else:
                            encoded.append(last[1])
                            i = last[0] + 1
                            node = self.trie_root
                            last = None
                    
                    if last is not None:
                        encoded.append(last[1])

                    # for i in range(len(bytes_)):
                    #     # start at root, or checking a node
                    #     if bytes([bytes_[i]]) in node.children:
                    #         # track where we are
                    #         # nodes.append(node)
                    #         if node.token_id is not None:
                    #             last_node = node
                    #         # go down
                    #         node = node.children[bytes([bytes_[i]])]
                    #     else:
                    #         # if not, the last value was the longest
                    #         # encoded.append(nodes[i-1].token_id)
                    #         encoded.append(last_node.token_id)
                    #         # and we go to the node for the unmatched byte, which should exist per construction
                    #         node = self.trie_root.children[bytes([bytes_[i-1]])]
                    #         # ... just make a "last node"?

        return encoded


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id


    def decode(self, ids: list[int]) -> str:
        result = b""
        for id_ in ids:
            result += self.vocab[id_]

        return result.decode("utf-8", errors="replace")



