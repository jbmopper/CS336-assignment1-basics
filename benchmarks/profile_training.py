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
from pathlib import Path

import numpy as np
import yaml

from cs336_basics import TransformerLM
from cs336_basics.training import (
    Trainer,
    setup_device,
    sync_device,
    load_tokens,
    print_model_summary,
)


# =============================================================================
# Model Configurations - loaded from configs/models.yaml
# =============================================================================

def load_model_configs() -> dict:
    """Load model configurations from configs/models.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


# Default data paths
DEFAULT_TRAIN_FILE = "tokenized/tinystories_train_fixed.npy"
DEFAULT_VALID_FILE = "tokenized/tinystories_valid_fixed.npy"


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
    
    # Print model info using shared utility
    print_model_summary(
        name=model_name,
        model_settings=model_config["model_settings"],
        batch_size=model_config["batch_size"],
        description=model_config.get("description"),
    )
    print(f"Device: {device}")
    print(f"Steps to profile: {num_steps}")
    
    # Create trainer
    checkpoint_dir = os.path.join("/tmp", f"profile_ckpt_{model_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer_config = get_trainer_config(model_config, device, checkpoint_dir)
    trainer = Trainer(TransformerLM, trainer_config, tokens, valid_tokens)
    
    # Model stats
    num_params = sum(p.numel() for p in trainer.model.parameters())
    d_head = model_config["model_settings"]["d_model"] // model_config["model_settings"]["num_heads"]
    tokens_per_step = model_config["batch_size"] * model_config["model_settings"]["context_length"]
    
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    
    # Validate/create trace dir
    if not os.path.exists(args.trace_dir):
        print(f"Creating output directory: {args.trace_dir}")
        os.makedirs(args.trace_dir, exist_ok=True)
    
    # Set up device using shared utility (or use specified device)
    if args.device:
        device = args.device
        print(f"Using specified device: {device}")
    else:
        device = setup_device(seed=args.seed, prefer_cuda=False)
    
    print(f"Output directory: {args.trace_dir}")
    
    # Load model configs from YAML
    model_configs = load_model_configs()
    model_a = model_configs["model_a"]
    model_b = model_configs["model_b"]
    
    # Load data using shared utility
    tokens, valid_tokens = load_tokens(DEFAULT_TRAIN_FILE, DEFAULT_VALID_FILE)
    
    # Profile
    results = []
    if args.model in ("A", "both"):
        r = profile_model(model_a, tokens, valid_tokens, device, args.trace_dir, args.num_steps)
        results.append(r)
    
    if args.model in ("B", "both"):
        r = profile_model(model_b, tokens, valid_tokens, device, args.trace_dir, args.num_steps)
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
