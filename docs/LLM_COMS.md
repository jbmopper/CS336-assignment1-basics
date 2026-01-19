# BPE Tokenizer Memory Analysis for OpenWebText

## Context

Training a BPE tokenizer on `owt_train.txt` (~10GB uncompressed) with a 32K vocabulary, per the CS336 assignment requirements (≤ 100GB RAM, ≤ 12 hours).

## Memory Analysis for BPE Training

### Key Data Structures in `train_bpe` (bpe.py)

1. **`pretoken_counts: Counter[str]`** — counts of unique pretokens
2. **`tokenized: dict[str, list[bytes]]`** — pretoken → list of byte tokens
3. **`pair_pretoken_map: dict[tuple[bytes, bytes], list[str]]`** — pair → list of pretokens containing it
4. **`paircount: Counter[tuple[bytes, bytes]]`** — pair counts

### Estimates for ~10GB Corpus

With ~10GB of text (the `owt_train.txt` file), the number of **unique pretokens**
is likely in the low‑millions. The exact memory footprint is dominated by
`pair_pretoken_map`, which can grow much faster than the unique pretoken count.

| Data Structure | Typical Scale |
|----------------|---------------|
| **pretoken_counts** | Low single‑digit GB |
| **tokenized** | A few GB |
| **paircount** | < 1 GB |
| **pair_pretoken_map** | **Potentially tens of GB (wildcard)** |

### Comparison: Full OpenWebText (~40GB)

For reference, if training on the full ~40GB OpenWebText:
- **Unique pretokens**: ~20-50M
- **Estimated memory**: ~30-60GB with Python overhead

### Verdict

**100GB RAM is likely sufficient** for the ~10GB `owt_train.txt` corpus, but it
is not guaranteed because `pair_pretoken_map` can dominate memory. The
implementation is optimized around *unique pretokens*, but still stores large
pair‑to‑pretoken lists in RAM.

---

## Tokenizing to .npy Output

### Current Approach (tokenize_corpus.py)

```python
text = f.read()  # Loads entire file into memory
tokens = tokenizer.encode(text)  # Creates Python list
arr = np.array(tokens, dtype=np.uint16)  # Converts to numpy
```

### Memory Concerns for 10GB File

| Stage | Memory Required |
|-------|-----------------|
| Text string | ~10 GB |
| Token list (Python ints) | ~20+ GB (Python list overhead) |
| Numpy array | ~2-4 GB (at 2 bytes/token) |
| **Peak total** | **~30-35 GB** |

This fits within 100GB but is inefficient.

### Why uint16 is Appropriate

- `uint16` range: 0–65,535
- Vocab size: 32,000
- Fits comfortably with room for larger vocabs up to 64K

### Recommended: Streaming Approach for Large Files

The `Tokenizer.encode_iterable()` method supports streaming tokenization. The challenge is handling output without knowing the final token count upfront.

#### Option A: Chunked Collection (Single Pass)

Collect tokens in numpy chunks, concatenate at end. Note: `np.concatenate`
creates a second full array, so peak memory can spike.

```python
import os
import numpy as np
from cs336_basics.bpe import load_bpe, Tokenizer

def tokenize_large_file(input_path, tokenizer_path, output_path, 
                        read_chunk_size=100_000_000, token_chunk_size=10_000_000):
    """Tokenize large file with controlled memory usage."""
    vocab, merges = load_bpe(tokenizer_path)
    tokenizer = Tokenizer(vocab, merges)
    
    def chunk_reader(path, size, special=b"<|endoftext|>"):
        """Yield chunks of text, splitting on document boundaries (byte-safe)."""
        buffer = b""
        with open(path, "rb") as f:
            while True:
                chunk = f.read(size)
                if not chunk:
                    break
                buffer += chunk
                parts = buffer.split(special)
                for part in parts[:-1]:
                    yield (part + special).decode("utf-8", errors="ignore")
                buffer = parts[-1]
            if buffer:
                yield buffer.decode("utf-8", errors="ignore")
    
    # Single pass: collect in numpy chunks
    chunks = []
    current_chunk = []
    
    for token_id in tokenizer.encode_iterable(chunk_reader(input_path, read_chunk_size)):
        current_chunk.append(token_id)
        if len(current_chunk) >= token_chunk_size:
            chunks.append(np.array(current_chunk, dtype=np.uint16))
            current_chunk = []
            print(f"Processed {sum(len(c) for c in chunks):,} tokens...")
    
    if current_chunk:
        chunks.append(np.array(current_chunk, dtype=np.uint16))
    
    arr = np.concatenate(chunks)
    np.save(output_path, arr)
    print(f"Saved {len(arr):,} tokens to {output_path}")
```

**Memory**: ~200MB per token chunk + final array (~5GB for 10GB input)

#### Option B: Direct Binary Write (Minimal Memory)

Write tokens directly to a binary file with buffered writes, then convert to
.npy using a memmap. Uses `chunk_reader` from Option A above.

```python
import os
import numpy as np
from cs336_basics.bpe import load_bpe, Tokenizer

def tokenize_minimal_memory(input_path, tokenizer_path, output_path, chunk_size=100_000_000):
    # chunk_reader defined in Option A above
    vocab, merges = load_bpe(tokenizer_path)
    tokenizer = Tokenizer(vocab, merges)
    
    temp_bin = output_path + ".bin"
    buffer = []
    count = 0
    with open(temp_bin, "wb") as out:
        for token_id in tokenizer.encode_iterable(chunk_reader(input_path, chunk_size)):
            buffer.append(token_id)
            if len(buffer) >= 5_000_000:
                arr = np.array(buffer, dtype=np.uint16)
                out.write(arr.tobytes(order="C"))
                count += len(arr)
                buffer.clear()
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            out.write(arr.tobytes(order="C"))
            count += len(arr)

    # Convert to .npy with memmaps
    token_count = os.path.getsize(temp_bin) // 2
    src = np.memmap(temp_bin, dtype=np.uint16, mode="r", shape=(token_count,))
    dst = np.lib.format.open_memmap(output_path, dtype=np.uint16, mode="w+", shape=(token_count,))
    dst[:] = src[:]
    del dst
    del src
    os.remove(temp_bin)
    print(f"Saved {count:,} tokens to {output_path}")
```

**Memory**: ~constant during tokenization, then ~5GB to load final array

#### Option C: Pre-allocated Memmap (Low RAM, More Disk)

Over-allocate based on file size estimate, truncate at end. Uses `chunk_reader` from Option A above.

```python
import os
import numpy as np
from cs336_basics.bpe import load_bpe, Tokenizer

def tokenize_memmap(input_path, tokenizer_path, output_path, chunk_size=100_000_000):
    # chunk_reader defined in Option A above
    vocab, merges = load_bpe(tokenizer_path)
    tokenizer = Tokenizer(vocab, merges)
    
    # Conservative estimate: assume ~2 bytes of text per token (compression ratio)
    file_size = os.path.getsize(input_path)
    max_tokens = file_size  # Very conservative upper bound
    
    temp_path = output_path + '.tmp'
    arr = np.memmap(temp_path, dtype=np.uint16, mode='w+', shape=(max_tokens,))
    
    idx = 0
    for token_id in tokenizer.encode_iterable(chunk_reader(input_path, chunk_size)):
        arr[idx] = token_id
        idx += 1
    
    arr.flush()
    del arr
    
    # Save as proper .npy without loading everything into RAM
    final_arr = np.memmap(temp_path, dtype=np.uint16, mode="r", shape=(idx,))
    out = np.lib.format.open_memmap(output_path, dtype=np.uint16, mode="w+", shape=(idx,))
    out[:] = final_arr[:]
    del out
    del final_arr
    os.remove(temp_path)
    print(f"Saved {idx:,} tokens")
```

**Memory**: ~constant (memmap handles paging), temp disk ~20GB, final ~5GB

### Special Considerations

1. **Document boundaries**: When chunking, avoid splitting in the middle of `<|endoftext|>` tokens
2. **UTF-8 boundaries**: Chunk in bytes and decode with `errors="ignore"` to avoid seeking in text mode
3. **Compression**: Consider saving with `np.savez_compressed()` if disk space is a concern (tokens compress well)

---

## Summary

| Task | Memory Estimate | 100GB Sufficient? |
|------|-----------------|-------------------|
| BPE Training (10GB corpus) | 20-80 GB (depends on `pair_pretoken_map`) | ⚠️ Likely |
| Tokenization (naive, current code) | 30-35 GB | ✅ Yes |
| Tokenization (chunked collection) | ~5-10 GB + concat spike | ✅ Yes |
| Tokenization (binary write) | ~constant + final copy | ✅ Yes |
| Tokenization (memmap) | ~constant RAM, ~20GB temp disk | ✅ Yes |

All operations should complete comfortably within the 100GB RAM budget.

### Recommendation

For a 10GB file within 100GB RAM:
- **Naive approach** is simplest but wasteful
- **Binary write + memmap** is a good default for large files
- **Memmap pre‑allocation** is best if RAM is tight and disk is ample
