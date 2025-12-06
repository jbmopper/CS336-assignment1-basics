import time
import json
import pickle
from cs336_basics.bpe import train_bpe, save_bpe, load_bpe

# Path to TinyStories (check where it is on your system)
# On course machines it's at /data, otherwise download per README
input_path = "./data/TinyStoriesV2-GPT4-train.txt"  

start_time = time.time()
vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"]
)
elapsed = time.time() - start_time
print(f"Training took: {elapsed:.2f} seconds")

output_path = "./tokenziers/tinystories/"
save_bpe(output_path, vocab, merges)
print(f"Saved to {output_path}.")

# Save vocab and merges
# with open("vocab.json", "w") as f:
#     # Convert bytes keys to strings for JSON serialization
#     json.dump({k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}, f)
# 
# # with open("merges.txt", "w") as f:
# #    for m1, m2 in merges:
# #        f.write(f"{m1.decode('utf-8', errors='replace')} {m2.decode('utf-8', errors='replace')}\n")
# with open("merges.pkl", "wb") as f:
#     pickle.dump(merges, f)
# 
# Find longest token
longest_token = max(vocab.values(), key=len)
print(f"Longest token: {longest_token} (length: {len(longest_token)} bytes)")