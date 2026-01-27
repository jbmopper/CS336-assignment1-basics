from cs336_basics.training import Trainer
from cs336_basics import TransformerLM
import numpy as np
import torch.utils.benchmark as benchmark
import torch
from itertools import product
import json
from datetime import datetime
import argparse

def main():
    # scaled_dot_product_attention()
    # the sdpa function take sthe Q, K, and V projection tensors
    # these are the inputs multiplied by the concatenated weights
    # each being [B, h, S, d_head]
    # the mask is a [B, h, S, S] tensor of booleans
    # the output is a [B, h, S, d_head] tensor

    # B can be varied to see how throughput scales... 4, 8, 16, 32, 64
    # d_head can be 32, 64, 96, 128, ... bigger?
    # h can be 4, 8, 16, 32
    # S can be 128, 256, 384, 512, 640, 768, 896, 1024
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--smol", action="store_true", help="smol run")
    parser.add_argument("--data", action="store", default="tinystories")
    args = parser.parse_args()

    if args.smol:
        Bs = [8, 32]
        d_heads = [64, 128]
        hs = [8]
        Ss = [256, 512, 1024]
        # Bs = [48, 64, 80]
        # d_heads = [32, 64, 96, 128]
        # hs = [4, 8, 16, 32, 64]
        # Ss = [384, 512, 640, 768, 896]
    else:
        Bs = [4, 8, 16, 32, 48, 64, 80]
        d_heads = [32, 64, 96, 128]
        hs = [4, 8, 16, 32, 64]
        Ss = [128, 256, 384, 512, 640, 768, 896, 1024]

    device = "mps"
    configs = [
        {"B": B, "d_head": d_head, "h": h, "S": S}
        for B, d_head, h, S in product(Bs, d_heads, hs, Ss)
    ]

    if args.data == "tinystories":
        config = {
            "train_file": "tokenized/tinystories_train_fixed.npy",
            "valid_file": "tokenized/tinystories_valid_fixed.npy",
        }
    else:
        raise ValueError(f"Unknown data: {args.data}")

    tokens = np.load(config["train_file"], mmap_mode='r')
    valid_tokens = np.load(config["valid_file"], mmap_mode='r')
    config = {
        # 1. Model Architecture (passed to your model class)
        "model_settings": {
            "vocab_size": 10000,      # Example value
            "d_model": 512,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 2048,
            "context_length": 512,
            "rope_theta": 10000.0,
        },
        
        # 2. Training Infrastructure
        "device": "mps",              # "cpu", "cuda", or "mps"
        "checkpoint_dir": "/dev/null",
        
        # 3. Hyperparameters
        "batch_size": 32,
        "num_iters": 5000,
        "eval_every": 500,
        "gradient_clip": 1.0,
    }
    config["vocab_size"] = 10000
    trainer = Trainer(TransformerLM, config, tokens, valid_tokens)

    
    results = []
    oom_configs = []
    for config in configs:
        try:
            label = "training"
            sublabel = "B={B}, d_head={d_head}, h={h}, S={S}".format(**config) 


            print(f"Running benchmark for {config['B']} batches x {config['d_head']} head dimension x {config['h']} heads x {config['S']} sequence length")
            torch.mps.synchronize()
            results.append(benchmark.Timer(
                stmt="scaled_dot_product_attention(Q, K, V, mask); torch.mps.synchronize()",
                globals={
                    "Q": Q, "K": K, "V": V, "mask": mask,
                    "torch": torch,
                    "scaled_dot_product_attention": scaled_dot_product_attention,
                },
                label=label,
                sub_label=sublabel,
                description=f"Seq Len {config['S']}",
            ).blocked_autorange(min_run_time=2))
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM for {config}, skipping...")
                oom_configs.append(config)
                torch.mps.empty_cache()  # Clear MPS memory cache
            else:
                raise
        finally:
            # Explicitly delete tensors and clear cache between configs
            del Q, K, V, mask
            torch.mps.empty_cache()

    rows = []
    for m in results:
        rows.append({
            "label": m.label,
            "sublabel": m.sub_label,
            "median_s": m.median,   # seconds
            "mean_s": m.mean,
            "iqr_s": m.iqr,
            "num_runs": len(m.times),
        })

    filename = f"micro_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(rows, f)
        print(f"Saved results to {filename}")   


    compare = benchmark.Compare(results)
    compare.colorize()
    compare.print()

if __name__ == "__main__":
    main()

""" for larger bencmarking
config = {}

# files exist
config["train_file"] = "tokenized/tinystories_train_fixed.npy"
config["valid_file"] = "tokenized/tinystories_valid_fixed.npy"

# Load tokenized data
tokens = np.load(config["train_file"], mmap_mode='r')
print(f"Training tokens loaded from {config['train_file']}, shape {tokens.shape}.")

valid_tokens = np.load(config["valid_file"], mmap_mode='r')
print(f"Validation tokens loaded from {config['valid_file']}, shape {valid_tokens.shape}.")
"""