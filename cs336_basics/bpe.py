import os
from typing import Any, BinaryIO
import multiprocessing
import regex as re
from collections import Counter

from torch import mul



def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {i: bytes([i]) for i in range(256)} # also need special tokens?
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = [] # merges.append((b"urg", b"bla"))
    pretoken_counts = Counter()
    paircount: Counter[tuple[bytes, bytes]] = Counter()

    with open(input_path, "rb") as f:
        num_processes =  8
        # boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        boundaries = find_chunk_boundaries(f, 8, b"<|endoftext|>")
        chunks = zip(boundaries[:-1], boundaries[1:])
        if __name__ == "__main__":
            with multiprocessing.Pool(...) as pool:
                chunked_counters = pool.map(lambda x: pretokenize(f, x, PAT), chunks)
        
    for cc in chunked_counters:
        pretoken_counts.update(cc)
    
    del pretoken_counts['<|']
    del pretoken_counts['|>']
    del pretoken_counts['endoftext']
        
    tokenized = { # need to actually merge bytes
        pretoken: [bytes([b]) for b in pretoken.encode("utf-8")]
        for pretoken in pretoken_counts.keys()
    }

    # first pass    
    for pretoken, b in tokenized.items():
        for i in range(len(b)-1): # skips len 1 b's
            paircount[(b[i], b[i+1])] += 1 * pretoken_counts[pretoken]


    while len(vocab) < vocab_size:
        # need the byte gumming here
        merge = max(paircount.items(), key=lambda kv: (kv[1], kv[0]))[0] # bot says this does it all...
        merges.append(merge)
        vocab[len(vocab)] = bytes(merge) # combines into one bytes object
        update_tokens(merge, tokenized)

    return vocab, merges



def pretokenize(
    file: BinaryIO,
    boundaries, # (int, int)
    pretokenizer: str
) -> Counter:
    pretoken_counts = Counter()
    file.seek(boundaries[0])
    chunk = file.read(boundaries[1] - boundaries[0]).decode("utf-8", errors="ignore")
    pretokens = re.finditer(pretokenizer, chunk)
    for pretoken in pretokens:
        pretoken_counts[pretoken.group(0)] += 1
    return pretoken_counts


def update_tokens( # updates token lists and pair counts
    merge: tuple[bytes, bytes],
    tokenized: dict[str, [bytes]],
    paircount: Counter,
    pretoken_count: Counter
) -> None:
    for pretoken, b in tokenized.items():
        updated_tokens = []
        i = 0
        while i < len(b):
            if i < len(b) - 1 and b[i] == merge[0] and b[i+i] == merge[1]:
                updated_tokens.append(bytes(merge))
                if i > 0:
                    paircount[(b[i-1], merge[0])] -= pretoken_count[pretoken] 
                if i + 2 < len(b):
                    paircount[(merge[1], b[i+2])] -= pretoken_count[pretoken]
                i += 2
            else:
                updated_tokens.append(b[i])
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