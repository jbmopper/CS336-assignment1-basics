"""Profile training steps for Model A and Model B.

Uses simple timing-based profiling that works reliably on MPS.
The Metal System Trace from xctrace provides the detailed GPU profiling.

Usage:
    uv run python -m benchmarks.profile_training --model A
    uv run python -m benchmarks.profile_training --model B  
    uv run python -m benchmarks.profile_training --model both
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch

from cs336_basics import TransformerLM
from cs336_basics.training import Trainer


# =============================================================================
# Model Configurations (same as train_compare.py)
# =============================================================================

MODEL_A_CONFIG = {
    "name": "Model_A_wide_attn",
    "description": "Wide attention, low FFN ratio (1.6x), d_head=64",
    "batch_size": 64,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 640,
        "num_heads": 10,  # d_head = 64
        "num_layers": 10,
        "d_ff": 1024,
        "context_length": 256,
        "rope_theta": 10000.0,
    },
}

MODEL_B_CONFIG = {
    "name": "Model_B_standard_ffn",
    "description": "Standard FFN ratio (4.5x), d_head=32 (potential perf issue)",
    "batch_size": 48,
    "model_settings": {
        "vocab_size": 10000,
        "d_model": 384,
        "num_heads": 12,  # d_head = 32
        "num_layers": 12,
        "d_ff": 1728,
        "context_length": 256,
        "rope_theta": 10000.0,
    },
}


def get_trainer_config(model_config: dict, device: str, checkpoint_dir: str) -> dict:
    """Create a minimal trainer config for profiling (no W&B)."""
    return {
        "model_settings": model_config["model_settings"],
        "device": device,
        "batch_size": model_config["batch_size"],
        "num_iters": 1000,
        "eval_every": 999999,
        "gradient_clip": 1.0,
        "scheduler_lr_max": 1e-3,
        "scheduler_lr_min": 1e-4,
        "scheduler_warmup_iters": 100,
        "scheduler_cos_iters": 1000,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_add_timestamp": False,
    }


def load_data():
    """Load tokenized training data."""
    train_file = "tokenized/tinystories_train_fixed.npy"
    valid_file = "tokenized/tinystories_valid_fixed.npy"

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training data not found: {train_file}\nRun from repository root.")

    tokens = np.load(train_file, mmap_mode='r')
    valid_tokens = np.load(valid_file, mmap_mode='r')
    print(f"Loaded {len(tokens):,} train tokens, {len(valid_tokens):,} valid tokens")
    return tokens, valid_tokens


def sync_device(device: str):
    """Synchronize device for accurate timing."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def profile_model(
    model_config: dict,
    tokens: np.ndarray,
    valid_tokens: np.ndarray,
    device: str,
    trace_dir: str,
    num_steps: int,
):
    """Profile a model with timing measurements."""
    model_name = model_config["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 60)
    print(f"Profiling: {model_name}")
    print(f"Description: {model_config['description']}")
    print(f"Device: {device}")
    print(f"Steps: {num_steps}")
    print("=" * 60)
    
    # Create trainer
    checkpoint_dir = os.path.join("/tmp", f"profile_ckpt_{model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer_config = get_trainer_config(model_config, device, checkpoint_dir)
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    # Model stats
    num_params = sum(p.numel() for p in trainer.model.parameters())
    d_head = model_config["model_settings"]["d_model"] // model_config["model_settings"]["num_heads"]
    tokens_per_step = model_config["batch_size"] * model_config["model_settings"]["context_length"]
    
    print(f"Parameters: {num_params / 1e6:.1f}M")
    print(f"d_head: {d_head}")
    print(f"Tokens/step: {tokens_per_step:,}")
    
    # Warmup
    warmup_steps = 5
    print(f"\nWarmup ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        trainer._train_step(i)
    sync_device(device)
    
    # Profile with timing
    print(f"\nProfiling {num_steps} steps...")
    step_times = []
    
    total_start = time.perf_counter()
    for step in range(num_steps):
        step_start = time.perf_counter()
        trainer._train_step(step + warmup_steps)
        sync_device(device)
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        
        if (step + 1) % 10 == 0:
            avg_so_far = sum(step_times) / len(step_times)
            print(f"  Step {step + 1}/{num_steps}: {step_time*1000:.1f}ms (avg: {avg_so_far*1000:.1f}ms)")
    
    total_time = time.perf_counter() - total_start
    
    # Statistics
    step_times_ms = [t * 1000 for t in step_times]
    avg_ms = sum(step_times_ms) / len(step_times_ms)
    min_ms = min(step_times_ms)
    max_ms = max(step_times_ms)
    sorted_times = sorted(step_times_ms)
    p50_ms = sorted_times[len(sorted_times) // 2]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    tokens_per_sec = tokens_per_step / (avg_ms / 1000)
    
    # Results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Total time:     {total_time:.1f}s")
    print(f"Avg step time:  {avg_ms:.1f}ms")
    print(f"Min step time:  {min_ms:.1f}ms")
    print(f"Max step time:  {max_ms:.1f}ms")
    print(f"P50 step time:  {p50_ms:.1f}ms")
    print(f"P95 step time:  {p95_ms:.1f}ms")
    print(f"Throughput:     {tokens_per_sec:.0f} tokens/sec")
    
    # Save summary
    summary_file = os.path.join(trace_dir, f"{model_name}_{timestamp}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Description: {model_config['description']}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Parameters: {num_params / 1e6:.1f}M\n")
        f.write(f"d_head: {d_head}\n")
        f.write(f"Batch size: {model_config['batch_size']}\n")
        f.write(f"Tokens/step: {tokens_per_step}\n")
        f.write(f"Steps profiled: {num_steps}\n")
        f.write(f"\n")
        f.write(f"Total time:     {total_time:.1f}s\n")
        f.write(f"Avg step time:  {avg_ms:.1f}ms\n")
        f.write(f"Min step time:  {min_ms:.1f}ms\n")
        f.write(f"Max step time:  {max_ms:.1f}ms\n")
        f.write(f"P50 step time:  {p50_ms:.1f}ms\n")
        f.write(f"P95 step time:  {p95_ms:.1f}ms\n")
        f.write(f"Throughput:     {tokens_per_sec:.0f} tokens/sec\n")
        f.write(f"\nStep times (ms):\n")
        for i, t in enumerate(step_times_ms):
            f.write(f"  {i}: {t:.1f}\n")
    
    print(f"\nSummary saved: {summary_file}")
    
    # Cleanup
    del trainer.model, trainer.optimizer, trainer
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "model_name": model_name,
        "summary_file": summary_file,
        "avg_ms": avg_ms,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Model A and/or Model B")
    parser.add_argument("--model", choices=["A", "B", "both"], default="both",
                        help="Which model(s) to profile")
    parser.add_argument("--trace-dir", default="/Volumes/slab_storage/traces",
                        help="Directory to save results")
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Number of training steps to profile")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not specified)")
    args = parser.parse_args()
    
    # Validate/create trace dir
    if not os.path.exists(args.trace_dir):
        print(f"Creating output directory: {args.trace_dir}")
        os.makedirs(args.trace_dir, exist_ok=True)
    
    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Device: {device}")
    print(f"Output directory: {args.trace_dir}")
    
    # Load data
    tokens, valid_tokens = load_data()
    
    # Profile
    results = []
    if args.model in ("A", "both"):
        r = profile_model(MODEL_A_CONFIG, tokens, valid_tokens, device, args.trace_dir, args.num_steps)
        results.append(r)
    
    if args.model in ("B", "both"):
        r = profile_model(MODEL_B_CONFIG, tokens, valid_tokens, device, args.trace_dir, args.num_steps)
        results.append(r)
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        a, b = results[0], results[1]
        speedup = b["avg_ms"] / a["avg_ms"]
        throughput_ratio = a["tokens_per_sec"] / b["tokens_per_sec"]
        print(f"Model A: {a['avg_ms']:.1f}ms/step, {a['tokens_per_sec']:.0f} tok/s")
        print(f"Model B: {b['avg_ms']:.1f}ms/step, {b['tokens_per_sec']:.0f} tok/s")
        print(f"Model A is {speedup:.2f}x faster per step")
        print(f"Model A has {throughput_ratio:.2f}x higher throughput")
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("\nThe detailed GPU profiling comes from xctrace (Metal System Trace).")
    print("This script provides timing data; xctrace shows what the GPU is doing.")


if __name__ == "__main__":
    main()
