# Code Improvements

Suggested improvements to the codebase. Each section describes the problem and solution approach.

---

## BPE Encoder Optimization

### Problem
The `Tokenizer.encode()` method is slow. For each pretoken, it iterates through ALL merges (~10K) checking if each one applies:

```python
for merge0, merge1 in self.merges:  # O(num_merges)
    # scan tokens looking for this specific merge
```

This is O(num_merges × pretoken_length) per pretoken.

### Solution
Build a merge rank lookup dict at initialization:

```python
self.merge_rank = {merge: i for i, merge in enumerate(self.merges)}
```

Then change the merge loop to:
1. Find all adjacent pairs in the current token list
2. Look up which pairs have merges (O(1) dict lookup)
3. Apply the merge with the lowest rank (earliest in merge list)
4. Repeat until no mergeable pairs remain

This reduces the inner loop from O(num_merges) to O(num_pairs).

### Files to modify
- `cs336_basics/bpe.py`: Add `merge_rank` dict in `Tokenizer.__init__`, rewrite merge loop in `encode()`

### Testing
Run `uv run pytest tests/test_tokenizer.py` - all tests should still pass.

---

## Tokenizer Save/Load Bug

### Problem
The current `from_files()` method has two bugs:

1. **Encoding mismatch:** Vocab is saved with `decode("utf-8")` but bytes 128-255 aren't valid UTF-8 sequences. They get replaced with `�` and can't be recovered.

2. **Special token check:** Compares string to bytes (`special_token not in vocab.values()`) which always fails in Python 3.

### Solution
Use `latin-1` encoding which maps bytes 0-255 to characters 0-255 directly:

```python
# Save
json.dump({k: v.decode("latin-1") for k, v in vocab.items()}, f)

# Load  
vocab = {int(k): v.encode("latin-1") for k, v in vocab_raw.items()}
```

Fix the special token check:
```python
if special_token.encode("utf-8") not in vocab.values():
```

### Files to modify
- `cs336_basics/bpe.py`: Fix `from_files()` and add `save()` method

---

## Wandb Integration

### Setup
Wandb is already in dependencies. One-time login:
```bash
uv run wandb login
```

### Usage pattern
```python
import wandb

wandb.init(project="cs336-training", config=config_dict)

# In training loop:
wandb.log({"train/loss": loss, "train/lr": lr}, step=iteration)

# At end:
wandb.finish()
```

### Useful metrics to log
- `train/loss` - training loss each step
- `train/perplexity` - exp(loss)
- `train/lr` - current learning rate
- `eval/val_loss` - validation loss (periodically)
- `eval/val_perplexity` - validation perplexity

---

## Perplexity

Standard language model metric. Lower is better.

```python
import math
perplexity = math.exp(loss.item())
```

Intuition: "The model is as uncertain as if choosing uniformly among X options" where X is perplexity.

| Perplexity | Meaning |
|------------|---------|
| vocab_size | Random (untrained) |
| 100-500 | Starting to learn |
| 20-50 | Decent small model |
| 10-20 | Good |

---

## Progress Bars with tqdm

```python
from tqdm.auto import tqdm

for it in tqdm(range(max_iters), desc="Training"):
    # ...
    pbar.set_postfix({"loss": f"{loss:.4f}"})
```

The `.auto` version renders nicely in both terminal and Jupyter.

